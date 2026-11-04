/// The longest prefix of `s` that fits in `max_bytes` and ends on a char boundary.
pub fn prefix_within(s: &str, max_bytes: usize) -> &str {
    (0..=s.len().min(max_bytes))
        .rev()
        .find_map(|end| s.get(..end))
        .unwrap_or_default()
}
