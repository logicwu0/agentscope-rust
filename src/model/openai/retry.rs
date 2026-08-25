//! Retry policy and `Retry-After` handling.

use std::time::{Duration, SystemTime};

use reqwest::header::{HeaderMap, RETRY_AFTER};

use super::super::ModelError;

/// Exponential-backoff policy for retryable model requests.
///
/// `max_retries` counts attempts after the initial request. Provider
/// `Retry-After` values are honored but capped by `max_delay`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
    max_retries: u32,
    initial_delay: Duration,
    max_delay: Duration,
}

impl RetryPolicy {
    /// Creates a policy with the requested number of retries.
    #[must_use]
    pub const fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            initial_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(10),
        }
    }

    /// Disables automatic retries.
    #[must_use]
    pub const fn disabled() -> Self {
        Self::new(0)
    }

    /// Sets the delay before the first retry.
    #[must_use]
    pub const fn with_initial_delay(mut self, initial_delay: Duration) -> Self {
        self.initial_delay = initial_delay;
        self
    }

    /// Sets the maximum delay between attempts.
    #[must_use]
    pub const fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Returns the maximum number of attempts after the initial request.
    #[must_use]
    pub const fn max_retries(self) -> u32 {
        self.max_retries
    }

    /// Returns the delay before the first retry.
    #[must_use]
    pub const fn initial_delay(self) -> Duration {
        self.initial_delay
    }

    /// Returns the maximum delay between attempts.
    #[must_use]
    pub const fn max_delay(self) -> Duration {
        self.max_delay
    }

    pub(super) fn can_retry(self, retries: u32, error: &ModelError) -> bool {
        retries < self.max_retries && error.retryable
    }

    pub(super) async fn wait(self, retry_number: u32, retry_after: Option<Duration>) {
        let delay = self.delay(retry_number, retry_after);
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
    }

    fn delay(self, retry_number: u32, retry_after: Option<Duration>) -> Duration {
        let exponential = self
            .initial_delay
            .saturating_mul(1_u32.checked_shl(retry_number).unwrap_or(u32::MAX));
        retry_after.unwrap_or(exponential).min(self.max_delay)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(2)
    }
}

pub(super) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    parse_retry_after(headers.get(RETRY_AFTER)?.to_str().ok()?, SystemTime::now())
}

fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()
        .map(|time| time.duration_since(now).unwrap_or(Duration::ZERO))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::{RetryPolicy, parse_retry_after};

    #[test]
    fn exponential_delay_is_capped() {
        let policy = RetryPolicy::new(5)
            .with_initial_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_millis(250));

        assert_eq!(policy.delay(0, None), Duration::from_millis(100));
        assert_eq!(policy.delay(1, None), Duration::from_millis(200));
        assert_eq!(policy.delay(2, None), Duration::from_millis(250));
        assert_eq!(
            policy.delay(0, Some(Duration::from_secs(30))),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn retry_after_supports_seconds_and_http_dates() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let later = now + Duration::from_secs(12);
        let date = httpdate::fmt_http_date(later);

        assert_eq!(parse_retry_after("7", now), Some(Duration::from_secs(7)));
        assert_eq!(parse_retry_after(&date, now), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after("invalid", now), None);
    }
}
