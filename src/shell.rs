/// Both the agent spawn and the claude-code install probe must resolve PATH
/// the same way — otherwise the probe says "installed" while the agent can't
/// find the binary, or vice versa.
pub fn user_login_shell() -> String {
    std::env::var("USER_SHELL").unwrap_or_else(|_| "/bin/sh".to_string())
}
