//! Eviction policy for idle claude residents kept warm between turns. Pure:
//! the caller measures memory (`sys`) and kills the victim (`main.rs`).
//!
//! With `n` idle residents their TTL is 48h while `n <= 2`, then halves at a
//! growing rate: `48h * 2^-((n-2)^0.735)` (3 -> 24h, 15 -> 30m). Memory scales
//! `n`: `n_eff = idle footprint / (RAM / 40)`. A single idle resident never
//! expires.
//!
//! A pinned chat's resident gets `PINNED_TTL_FACTOR` x that TTL (still capped
//! at 48h) and is picked as victim only when every idle resident is pinned —
//! the unpinned crowd goes first, pressure included. An always-on chat's
//! resident is never picked; it still counts toward `n_eff`.

use std::time::Duration;
use tokio::time::Instant;

const MAX_TTL: Duration = Duration::from_secs(48 * 3600);
/// An idle resident this share of RAM counts as one.
const RAM_SHARE: f64 = 1.0 / 40.0;
/// How much longer a pinned chat's resident stays warm.
const PINNED_TTL_FACTOR: f64 = 8.0;

pub struct IdleResident {
    pub topic: String,
    /// End of its latest turn: when it emitted its idle edge.
    pub idle_since: Instant,
    pub pid: Option<u32>,
    /// Memory its whole process tree (MCP servers and browsers included)
    /// would free if killed.
    pub footprint_kb: Option<u64>,
    /// `chats.pinned_at` is set (`state.rs`).
    pub pinned: bool,
    /// `chats.always_on` is set (`state.rs`).
    pub always_on: bool,
}

pub struct Memory {
    pub total_kb: u64,
    /// The OS reports memory pressure (`sys::memory`).
    pub pressure: bool,
}

#[derive(Debug, PartialEq)]
pub enum Decision {
    Evict(Eviction),
    WaitUntil(Instant),
    KeepAll,
}

#[derive(Debug, PartialEq)]
pub struct Eviction {
    /// Index into the idle residents.
    pub victim: usize,
    pub n_eff: f64,
    pub ttl: Duration,
    pub age: Duration,
    pub pressure: bool,
}

pub fn ttl(n_eff: f64) -> Duration {
    if n_eff <= 2.0 {
        return MAX_TTL;
    }
    MAX_TTL.mul_f64(2f64.powf(-(n_eff - 2.0).powf(0.735)))
}

fn ttl_for(r: &IdleResident, n_eff: f64) -> Duration {
    let base = ttl(n_eff);
    if r.pinned {
        base.mul_f64(PINNED_TTL_FACTOR).min(MAX_TTL)
    } else {
        base
    }
}

/// Idle footprint in units of RAM/40, or the idle count when either is unknown.
pub fn n_eff(idle: &[IdleResident], memory: Option<&Memory>) -> f64 {
    let footprint_kb: Option<u64> = idle.iter().map(|r| r.footprint_kb).sum();
    match (footprint_kb, memory) {
        (Some(kb), Some(m)) if m.total_kb > 0 => kb as f64 / (m.total_kb as f64 * RAM_SHARE),
        _ => idle.len() as f64,
    }
}

/// The oldest unpinned idle resident goes once it outlives its TTL, or at once
/// under memory pressure; the last one standing stays.
pub fn choose(idle: &[IdleResident], memory: Option<&Memory>, now: Instant) -> Decision {
    if idle.len() < 2 {
        return Decision::KeepAll;
    }
    let n_eff = n_eff(idle, memory);
    let pressure = memory.is_some_and(|m| m.pressure);
    let age = |r: &IdleResident| now.saturating_duration_since(r.idle_since);
    let victim = idle
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.always_on && (pressure || age(r) >= ttl_for(r, n_eff)))
        .min_by_key(|(_, r)| (r.pinned, r.idle_since));
    match victim {
        Some((victim, r)) => Decision::Evict(Eviction {
            victim,
            n_eff,
            ttl: ttl_for(r, n_eff),
            age: age(r),
            pressure,
        }),
        None => idle
            .iter()
            .filter(|r| !r.always_on)
            .map(|r| r.idle_since + ttl_for(r, n_eff))
            .min()
            .map_or(Decision::KeepAll, Decision::WaitUntil),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: f64 = 3600.0;
    const GB_KB: u64 = 1024 * 1024;

    fn hours(d: Duration) -> f64 {
        d.as_secs_f64() / HOUR
    }

    fn idle(ages_h: &[f64], footprint_kb: Option<u64>, now: Instant) -> Vec<IdleResident> {
        ages_h
            .iter()
            .enumerate()
            .map(|(i, &h)| IdleResident {
                topic: format!("chat-{i}"),
                idle_since: now - Duration::from_secs_f64(h * HOUR),
                pid: None,
                footprint_kb,
                pinned: false,
                always_on: false,
            })
            .collect()
    }

    fn roomy(total_gb: u64) -> Memory {
        Memory {
            total_kb: total_gb * GB_KB,
            pressure: false,
        }
    }

    #[test]
    fn ttl_by_idle_count() {
        let expected_h = [
            (1, 48.0),
            (2, 48.0),
            (3, 24.0),
            (4, 15.2),
            (5, 10.1),
            (10, 1.96),
            (15, 0.50),
            (20, 0.145),
            (30, 0.0157),
        ];
        for (n, h) in expected_h {
            let got = hours(ttl(f64::from(n)));
            assert!((got - h).abs() / h < 0.02, "n={n}: {got}h, expected ~{h}h");
        }
    }

    #[test]
    fn ttl_by_memory() {
        let now = Instant::now() + Duration::from_secs(100 * 3600);
        // 30 x 300MB on 64GB: n_eff 5.5 -> ~8h.
        let heavy = idle(&[0.0; 30], Some(300 * 1024), now);
        let n = n_eff(&heavy, Some(&roomy(64)));
        assert!((n - 5.49).abs() < 0.01, "{n}");
        assert!((hours(ttl(n)) - 8.45).abs() < 0.1);
        // 10 x 100MB on 64GB is under two RAM/40 shares: full 48h.
        let light = idle(&[0.0; 10], Some(100 * 1024), now);
        assert_eq!(ttl(n_eff(&light, Some(&roomy(64)))), MAX_TTL);
        // Unknown footprint falls back to the count.
        let unmeasured = idle(&[0.0; 10], None, now);
        assert_eq!(n_eff(&unmeasured, Some(&roomy(64))), 10.0);
        assert_eq!(n_eff(&light, None), 10.0);
    }

    #[test]
    fn a_single_idle_resident_never_expires() {
        let now = Instant::now() + Duration::from_secs(1000 * 3600);
        let heavy_and_pressed = Memory {
            total_kb: GB_KB,
            pressure: true,
        };
        let one = idle(&[999.0], Some(10 * GB_KB), now);
        assert_eq!(
            choose(&one, Some(&heavy_and_pressed), now),
            Decision::KeepAll
        );
        assert_eq!(choose(&[], None, now), Decision::KeepAll);
    }

    #[test]
    fn evicts_the_oldest_only_past_its_ttl() {
        let now = Instant::now() + Duration::from_secs(100 * 3600);
        let three = idle(&[1.0, 23.0, 2.0], None, now);
        assert_eq!(
            choose(&three, None, now),
            Decision::WaitUntil(three[1].idle_since + Duration::from_secs(24 * 3600))
        );

        let past_ttl = idle(&[1.0, 25.0, 30.0], None, now);
        let Decision::Evict(e) = choose(&past_ttl, None, now) else {
            panic!("expected an eviction");
        };
        assert_eq!(e.victim, 2);
        assert_eq!(e.n_eff, 3.0);
        assert!(!e.pressure);
    }

    #[test]
    fn pinned_residents_go_last_and_on_a_longer_ttl() {
        let now = Instant::now() + Duration::from_secs(1000 * 3600);
        // 10 idle -> 1.96h unpinned, 15.7h pinned.
        let mut ten = idle(&[3.0; 10], None, now);
        ten[0].pinned = true;
        ten[0].idle_since = now - Duration::from_secs(20 * 3600);
        let Decision::Evict(e) = choose(&ten, None, now) else {
            panic!("expected an eviction");
        };
        assert_ne!(e.victim, 0, "the pinned one outlives the unpinned crowd");

        // All ten pinned: the oldest goes, but only once past the 15.7h TTL.
        for r in ten.iter_mut() {
            r.pinned = true;
        }
        let Decision::Evict(oldest) = choose(&ten, None, now) else {
            panic!("expected an eviction");
        };
        assert_eq!(oldest.victim, 0);
        assert!((hours(oldest.ttl) - 15.7).abs() < 0.1, "{:?}", oldest.ttl);

        ten[0].idle_since = now - Duration::from_secs(10 * 3600);
        assert!(matches!(choose(&ten, None, now), Decision::WaitUntil(_)));
    }

    #[test]
    fn pinned_ttl_is_still_capped_at_48h() {
        let now = Instant::now() + Duration::from_secs(1000 * 3600);
        let mut two = idle(&[49.0, 1.0], None, now);
        two[0].pinned = true;
        let Decision::Evict(e) = choose(&two, None, now) else {
            panic!("expected an eviction");
        };
        assert_eq!((e.victim, e.ttl), (0, MAX_TTL));
    }

    #[test]
    fn always_on_residents_are_never_evicted() {
        let now = Instant::now() + Duration::from_secs(1000 * 3600);
        let pressed = Memory {
            total_kb: GB_KB,
            pressure: true,
        };
        let mut three = idle(&[999.0, 998.0, 1.0], Some(GB_KB), now);
        three[0].always_on = true;
        three[1].always_on = true;
        let Decision::Evict(e) = choose(&three, Some(&pressed), now) else {
            panic!("expected an eviction");
        };
        assert_eq!(e.victim, 2);
        assert_eq!(e.n_eff, 120.0, "always-on residents still count");

        three[2].always_on = true;
        assert_eq!(choose(&three, Some(&pressed), now), Decision::KeepAll);
        assert_eq!(choose(&three, None, now), Decision::KeepAll);
    }

    #[test]
    fn memory_pressure_evicts_the_oldest_at_once() {
        let now = Instant::now() + Duration::from_secs(100 * 3600);
        let pressed = Memory {
            total_kb: 64 * GB_KB,
            pressure: true,
        };
        let two = idle(&[0.1, 0.2], Some(1024), now);
        let Decision::Evict(e) = choose(&two, Some(&pressed), now) else {
            panic!("expected an eviction");
        };
        assert_eq!(e.victim, 1);
        assert!(e.pressure);
    }
}
