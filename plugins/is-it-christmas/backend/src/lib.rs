// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! "Is it Christmas?" as a NeMo Helix community plugin, with a Rust backend.
//!
//! Helix never runs or imports this code. It reads a manifest from `/plugin.json`,
//! pins the UI bundle served at `/index.js`, and then proxies
//! `/apis/plugins/is-it-christmas/...` requests to the endpoint below.

use std::{
    cell::RefCell,
    panic,
    path::PathBuf,
    sync::{Arc, Once},
};

use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use chrono::{Datelike, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const PLUGIN_NAME: &str = "is-it-christmas";

/// A 640px copy of the original photo, compiled in so the server is self-contained.
const STEFAN_JPEG: &[u8] = include_bytes!("../assets/stefan_waving_christmas.jpeg");

/// An image in a form the Studio UI can use directly.
///
/// The Helix gateway only relays JSON and text as-is: any other content type is rewritten to
/// `application/octet-stream` and sent as an attachment, so a raw `image/jpeg` response would
/// download instead of rendering. Wrapping the image in JSON as a `data:` URL sidesteps that,
/// and Studio's content security policy allows `data:` images.
#[derive(Debug, PartialEq, Serialize)]
pub struct Picture {
    pub name: &'static str,
    pub content_type: &'static str,
    pub bytes: usize,
    pub data_url: String,
}

/// A 360px, 64-colour copy of the animated original (30 frames, loops forever).
/// The same figure, crying, for the NO page.
const STEFAN_SAD_JPEG: &[u8] = include_bytes!("../assets/stefan_sad.jpeg");

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mood {
    #[default]
    Happy,
    Sad,
}

#[derive(Deserialize)]
struct PictureQuery {
    /// `happy` (waving, the default) or `sad` (for the NO page).
    #[serde(default)]
    mood: Mood,
}

const STEFAN_GIF: &[u8] = include_bytes!("../assets/stefan_waving_animated_christmas.gif");

fn picture(content_type: &'static str, image: &[u8]) -> Picture {
    Picture {
        name: "Stefan",
        content_type,
        bytes: image.len(),
        data_url: format!("data:{content_type};base64,{}", STANDARD.encode(image)),
    }
}

pub fn stefan(mood: Mood) -> Picture {
    match mood {
        Mood::Happy => picture("image/jpeg", STEFAN_JPEG),
        Mood::Sad => picture("image/jpeg", STEFAN_SAD_JPEG),
    }
}

/// The animated version, shown on the error page. It is its own endpoint so the large image
/// never rides along inside the 500 body that the CLI and network tools print.
pub fn stefan_animated() -> Picture {
    picture("image/gif", STEFAN_GIF)
}

async fn stefan_endpoint(
    Path(_workspace): Path<String>,
    Query(query): Query<PictureQuery>,
) -> Json<Picture> {
    Json(stefan(query.mood))
}

async fn stefan_animated_endpoint(Path(_workspace): Path<String>) -> Json<Picture> {
    Json(stefan_animated())
}

/// Forces the answer for demos: set `CHRISTMAS_OVERRIDE=yes|no|auto` when starting the server.
/// `auto` (the default) follows the calendar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Override {
    #[default]
    Auto,
    Yes,
    No,
}

impl std::str::FromStr for Override {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Override::Auto),
            "yes" => Ok(Override::Yes),
            "no" => Ok(Override::No),
            other => Err(format!(
                "CHRISTMAS_OVERRIDE must be yes, no or auto, not {other:?}"
            )),
        }
    }
}

impl Override {
    /// The date to answer for. `Yes` pretends it is Christmas Day; `No` pretends it is not.
    pub fn apply(self, today: NaiveDate) -> NaiveDate {
        let christmas = NaiveDate::from_ymd_opt(today.year(), 12, 25);
        match (self, christmas) {
            (Override::Yes, Some(day)) => day,
            (Override::No, Some(day)) if today == day => day.pred_opt().unwrap_or(today),
            _ => today,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Where Helix can reach this server (used in the manifest's upstream and bundle URLs).
    pub public_url: String,
    /// Directory holding the built UI: `index.js` and `plugin.manifest.json`.
    pub bundle_dir: PathBuf,
    /// Forces the YES or NO answer; see [`Override`].
    pub answer_override: Override,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Answer {
    pub date: String,
    pub is_christmas: bool,
    pub days_until: i64,
    pub next_christmas: String,
    pub message: String,
    /// True when `CHRISTMAS_OVERRIDE` decided the answer instead of the calendar.
    pub forced: bool,
}

/// Answer the question for `today`. `None` only for dates at the edge of chrono's range.
pub fn answer(today: NaiveDate) -> Option<Answer> {
    let this_year = NaiveDate::from_ymd_opt(today.year(), 12, 25)?;
    let next = if today <= this_year {
        this_year
    } else {
        NaiveDate::from_ymd_opt(today.year() + 1, 12, 25)?
    };
    let days_until = (next - today).num_days();
    let message = match days_until {
        0 => "Yes! Merry Christmas.".to_string(),
        1 => "No, but it's tomorrow.".to_string(),
        n => format!("No. {n} days to go."),
    };
    Some(Answer {
        date: today.to_string(),
        is_christmas: days_until == 0,
        days_until,
        next_christmas: next.to_string(),
        message,
        forced: false,
    })
}

#[derive(Deserialize)]
struct AnswerQuery {
    /// The caller's local date as `YYYY-MM-DD`. Defaults to today in UTC.
    date: Option<String>,
}

fn bad_request(detail: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "detail": detail }))).into_response()
}

async fn is_it_christmas(
    State(config): State<Arc<Config>>,
    Path(_workspace): Path<String>,
    Query(query): Query<AnswerQuery>,
) -> Response {
    let today = match query.date {
        None => Utc::now().date_naive(),
        Some(raw) => match NaiveDate::parse_from_str(&raw, "%Y-%m-%d") {
            Ok(date) => date,
            Err(_) => return bad_request("date must be YYYY-MM-DD"),
        },
    };
    match answer(config.answer_override.apply(today)) {
        Some(mut answer) => {
            answer.forced = config.answer_override != Override::Auto;
            Json(answer).into_response()
        }
        None => bad_request("date is out of range"),
    }
}

thread_local! {
    /// Where the most recent panic on this thread happened, captured by the hook below.
    static LAST_PANIC_LOCATION: RefCell<Option<String>> = const { RefCell::new(None) };
}
static INSTALL_HOOK: Once = Once::new();

/// Keeps Rust's normal panic output on stderr, and also remembers where the panic
/// happened so the HTTP error can report it.
fn install_panic_hook() {
    INSTALL_HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if let Some(at) = info.location() {
                let place = format!("{}:{}:{}", at.file(), at.line(), at.column());
                LAST_PANIC_LOCATION.with(|slot| *slot.borrow_mut() = Some(place));
            }
            previous(info);
        }));
    });
}

#[derive(Debug, Serialize)]
pub struct PanicReport {
    /// The text Rust prints to stderr for a panic, including the RUST_BACKTRACE note.
    pub detail: String,
    pub kind: &'static str,
    pub thread: String,
    pub location: Option<String>,
    pub message: String,
}

/// Panic on purpose, the way real Rust code does: `unwrap()` on an `Err`. The panic is
/// caught so the server keeps serving, and reported exactly as it happened.
pub fn trigger_panic() -> PanicReport {
    install_panic_hook();
    let outcome = panic::catch_unwind(|| "ho ho ho".parse::<u32>().unwrap());
    let payload = outcome.expect_err("\"ho ho ho\" is not a number");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "Box<dyn Any>".to_string());
    let thread = std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_string();
    let location = LAST_PANIC_LOCATION.with(|slot| slot.borrow_mut().take());
    let detail = format!(
        "thread '{thread}' panicked at {}:\n{message}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace",
        location.as_deref().unwrap_or("<unknown>"),
    );
    PanicReport {
        detail,
        kind: "panic",
        thread,
        location,
        message,
    }
}

async fn panic_endpoint(Path(_workspace): Path<String>) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(trigger_panic())).into_response()
}

/// The manifest Helix registers. The bundle hash is computed from the file actually
/// served, so the pin cannot drift from the bytes. Without a built UI it is API-only.
fn manifest(config: &Config) -> Value {
    let base = config.public_url.trim_end_matches('/');
    let mut plugin = json!({
        "name": PLUGIN_NAME,
        "title": "Is it Christmas?",
        "endpoints": [
            {
                "method": "GET",
                "path": "/v2/workspaces/{workspace}/is-it-christmas",
                "access": "read",
                "upstream_url": format!("{base}/v1/workspaces/{{workspace}}/is-it-christmas"),
            },
            {
                "method": "GET",
                "path": "/v2/workspaces/{workspace}/panic",
                "access": "read",
                "upstream_url": format!("{base}/v1/workspaces/{{workspace}}/panic"),
            },
            {
                "method": "GET",
                "path": "/v2/workspaces/{workspace}/stefan",
                "access": "read",
                "upstream_url": format!("{base}/v1/workspaces/{{workspace}}/stefan"),
            },
            {
                "method": "GET",
                "path": "/v2/workspaces/{workspace}/stefan-animated",
                "access": "read",
                "upstream_url": format!("{base}/v1/workspaces/{{workspace}}/stefan-animated"),
            },
        ],
    });
    let script = std::fs::read(config.bundle_dir.join("index.js"));
    let sidecar = std::fs::read(config.bundle_dir.join("plugin.manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    if let (Ok(script), Some(sidecar)) = (script, sidecar) {
        plugin["bundle"] = json!({
            "url": format!("{base}/index.js"),
            "sha256": format!("{:x}", Sha256::digest(&script)),
            "sidecar": sidecar,
        });
        plugin["navigation"] = json!([{
            "label": "Is it Christmas?",
            "path": format!("/workspaces/{{workspace}}/plugin/{PLUGIN_NAME}"),
            "section": "Community",
        }]);
    }
    json!({ "schema_version": 1, "plugin": plugin })
}

async fn plugin_manifest(State(config): State<Arc<Config>>) -> Json<Value> {
    Json(manifest(&config))
}

async fn bundle(State(config): State<Arc<Config>>) -> Response {
    match std::fs::read(config.bundle_dir.join("index.js")) {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, "text/javascript"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Build the plugin web bundle first").into_response(),
    }
}

pub fn router(config: Config) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/plugin.json", get(plugin_manifest))
        .route("/index.js", get(bundle))
        .route(
            "/v1/workspaces/{workspace}/is-it-christmas",
            get(is_it_christmas),
        )
        .route("/v1/workspaces/{workspace}/panic", get(panic_endpoint))
        .route("/v1/workspaces/{workspace}/stefan", get(stefan_endpoint))
        .route(
            "/v1/workspaces/{workspace}/stefan-animated",
            get(stefan_animated_endpoint),
        )
        .with_state(Arc::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn day(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn christmas_day_is_yes() {
        let a = answer(day(2026, 12, 25)).unwrap();
        assert!(a.is_christmas);
        assert_eq!((a.days_until, a.next_christmas.as_str()), (0, "2026-12-25"));
    }

    #[test]
    fn christmas_eve_is_tomorrow() {
        let a = answer(day(2026, 12, 24)).unwrap();
        assert!(!a.is_christmas);
        assert_eq!(a.days_until, 1);
        assert_eq!(a.message, "No, but it's tomorrow.");
    }

    #[test]
    fn boxing_day_counts_to_next_year() {
        let a = answer(day(2026, 12, 26)).unwrap();
        assert_eq!(
            (a.days_until, a.next_christmas.as_str()),
            (364, "2027-12-25")
        );
    }

    #[test]
    fn leap_year_days_are_counted() {
        assert_eq!(answer(day(2024, 2, 29)).unwrap().days_until, 300);
    }

    fn temp_bundle(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("is-it-christmas-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
        let res = app
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn app_with(dir: PathBuf, answer_override: Override) -> Router {
        router(Config {
            public_url: "http://127.0.0.1:6010/".into(),
            bundle_dir: dir,
            answer_override,
        })
    }

    fn app(dir: PathBuf) -> Router {
        app_with(dir, Override::Auto)
    }

    #[test]
    fn override_yes_is_christmas_and_no_is_not_whatever_the_date() {
        let some_tuesday = day(2026, 11, 17);
        assert_eq!(Override::Yes.apply(some_tuesday), day(2026, 12, 25));
        assert_eq!(Override::No.apply(some_tuesday), some_tuesday);
        assert_eq!(Override::No.apply(day(2026, 12, 25)), day(2026, 12, 24));
        assert_eq!(Override::Auto.apply(day(2026, 12, 25)), day(2026, 12, 25));
    }

    #[test]
    fn override_parses_the_values_people_will_type_and_rejects_typos() {
        assert_eq!("YES".parse::<Override>(), Ok(Override::Yes));
        assert_eq!(" no ".parse::<Override>(), Ok(Override::No));
        assert_eq!("".parse::<Override>(), Ok(Override::Auto));
        assert!("maybe"
            .parse::<Override>()
            .unwrap_err()
            .contains("yes, no or auto"));
    }

    #[tokio::test]
    async fn the_override_decides_the_answer_and_says_so() {
        let url = "/v1/workspaces/default/is-it-christmas?date=2026-11-17";
        let (_, yes) = get_json(app_with(temp_bundle("yes"), Override::Yes), url).await;
        assert_eq!(
            (yes["is_christmas"].clone(), yes["forced"].clone()),
            (true.into(), true.into())
        );
        let (_, no) = get_json(app_with(temp_bundle("no"), Override::No), url).await;
        assert_eq!(
            (no["is_christmas"].clone(), no["forced"].clone()),
            (false.into(), true.into())
        );
        let (_, auto) = get_json(app(temp_bundle("auto")), url).await;
        assert_eq!(
            (auto["is_christmas"].clone(), auto["forced"].clone()),
            (false.into(), false.into())
        );
    }

    #[tokio::test]
    async fn the_sad_picture_is_a_different_jpeg() {
        let base = "/v1/workspaces/default/stefan";
        let (_, happy) = get_json(app(temp_bundle("happy")), base).await;
        let (_, sad) = get_json(app(temp_bundle("sad")), &format!("{base}?mood=sad")).await;
        assert_eq!(sad["content_type"], "image/jpeg");
        assert_ne!(happy["data_url"], sad["data_url"]);
        assert_eq!(sad["bytes"], STEFAN_SAD_JPEG.len());
    }

    #[test]
    fn stefan_is_a_real_jpeg_in_a_data_url() {
        let picture = stefan(Mood::Happy);
        let encoded = picture
            .data_url
            .strip_prefix("data:image/jpeg;base64,")
            .expect("a jpeg data url");
        let decoded = STANDARD.decode(encoded).expect("valid base64");
        assert_eq!(decoded, STEFAN_JPEG);
        assert_eq!(picture.bytes, decoded.len());
        assert_eq!(
            &decoded[..3],
            &[0xFF, 0xD8, 0xFF],
            "starts with the JPEG magic bytes"
        );
    }

    #[tokio::test]
    async fn stefan_endpoint_answers_json_the_gateway_will_relay() {
        let (status, body) =
            get_json(app(temp_bundle("stefan")), "/v1/workspaces/default/stefan").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["name"], "Stefan");
        assert_eq!(body["content_type"], "image/jpeg");
        assert!(body["data_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn animated_stefan_is_a_looping_gif_in_a_data_url() {
        let picture = stefan_animated();
        let encoded = picture
            .data_url
            .strip_prefix("data:image/gif;base64,")
            .expect("a gif data url");
        let decoded = STANDARD.decode(encoded).expect("valid base64");
        assert_eq!(decoded, STEFAN_GIF);
        assert_eq!(&decoded[..6], b"GIF89a");
        assert!(
            decoded.windows(11).any(|w| w == b"NETSCAPE2.0"),
            "has the infinite-loop extension"
        );
    }

    #[tokio::test]
    async fn animated_endpoint_is_separate_so_the_panic_body_stays_small() {
        let app = app(temp_bundle("animated"));
        let (status, body) = get_json(app.clone(), "/v1/workspaces/default/stefan-animated").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["content_type"], "image/gif");
        let (_, panic_body) = get_json(app, "/v1/workspaces/default/panic").await;
        assert!(
            panic_body.get("data_url").is_none(),
            "the 500 body carries no image"
        );
    }

    #[test]
    fn panic_report_is_in_rusts_own_words() {
        let report = trigger_panic();
        assert_eq!(
            report.message,
            "called `Result::unwrap()` on an `Err` value: ParseIntError { kind: InvalidDigit }"
        );
        assert!(report
            .location
            .as_deref()
            .is_some_and(|l| l.starts_with("src/lib.rs:")));
        assert!(report.detail.contains("panicked at src/lib.rs:"));
        assert!(report.detail.contains("RUST_BACKTRACE=1"));
    }

    #[tokio::test]
    async fn panic_endpoint_is_a_500_and_the_server_keeps_serving() {
        let app = app(temp_bundle("panic"));
        let (status, body) = get_json(app.clone(), "/v1/workspaces/default/panic").await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["kind"], "panic");
        assert!(body["detail"]
            .as_str()
            .unwrap()
            .contains("ParseIntError { kind: InvalidDigit }"));
        let (status, _) = get_json(app, "/health").await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn endpoint_uses_the_callers_date() {
        let (status, body) = get_json(
            app(temp_bundle("api")),
            "/v1/workspaces/default/is-it-christmas?date=2026-12-25",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["is_christmas"], true);
    }

    #[tokio::test]
    async fn endpoint_rejects_a_malformed_date() {
        let (status, body) = get_json(
            app(temp_bundle("bad")),
            "/v1/workspaces/default/is-it-christmas?date=tomorrow",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "date must be YYYY-MM-DD");
    }

    #[tokio::test]
    async fn manifest_without_a_built_ui_is_api_only() {
        let (_, body) = get_json(app(temp_bundle("none")), "/plugin.json").await;
        let plugin = &body["plugin"];
        assert_eq!(plugin["name"], PLUGIN_NAME);
        assert_eq!(
            plugin["endpoints"][0]["upstream_url"],
            "http://127.0.0.1:6010/v1/workspaces/{workspace}/is-it-christmas"
        );
        assert!(plugin.get("bundle").is_none() && plugin.get("navigation").is_none());
        assert_eq!(plugin["endpoints"].as_array().map(Vec::len), Some(4));
        assert_eq!(
            plugin["endpoints"][1]["path"],
            "/v2/workspaces/{workspace}/panic"
        );
    }

    #[tokio::test]
    async fn manifest_pins_the_bytes_it_serves() {
        let dir = temp_bundle("ui");
        std::fs::write(dir.join("index.js"), b"export const Root = () => null;").unwrap();
        std::fs::write(
            dir.join("plugin.manifest.json"),
            br#"{"contractVersion":"1.0.0"}"#,
        )
        .unwrap();
        let (_, body) = get_json(app(dir.clone()), "/plugin.json").await;
        let expected = format!("{:x}", Sha256::digest(b"export const Root = () => null;"));
        assert_eq!(body["plugin"]["bundle"]["sha256"], expected.as_str());
        assert_eq!(
            body["plugin"]["bundle"]["url"],
            "http://127.0.0.1:6010/index.js"
        );
        assert_eq!(
            body["plugin"]["bundle"]["sidecar"]["contractVersion"],
            "1.0.0"
        );
        assert_eq!(
            body["plugin"]["navigation"][0]["path"],
            "/workspaces/{workspace}/plugin/is-it-christmas"
        );
    }
}
