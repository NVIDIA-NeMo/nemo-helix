// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! "Is it Christmas?" as a NeMo Helix community plugin, with a Rust backend.
//!
//! Helix never runs or imports this code. It reads a manifest from `/plugin.json`,
//! pins the UI bundle served at `/index.js`, and then proxies
//! `/apis/plugins/is-it-christmas/...` requests to the endpoint below.

use std::{path::PathBuf, sync::Arc};

use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use chrono::{Datelike, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const PLUGIN_NAME: &str = "is-it-christmas";

#[derive(Clone, Debug)]
pub struct Config {
    /// Where Helix can reach this server (used in the manifest's upstream and bundle URLs).
    pub public_url: String,
    /// Directory holding the built UI: `index.js` and `plugin.manifest.json`.
    pub bundle_dir: PathBuf,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Answer {
    pub date: String,
    pub is_christmas: bool,
    pub days_until: i64,
    pub next_christmas: String,
    pub message: String,
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

async fn is_it_christmas(Path(_workspace): Path<String>, Query(query): Query<AnswerQuery>) -> Response {
    let today = match query.date {
        None => Utc::now().date_naive(),
        Some(raw) => match NaiveDate::parse_from_str(&raw, "%Y-%m-%d") {
            Ok(date) => date,
            Err(_) => return bad_request("date must be YYYY-MM-DD"),
        },
    };
    match answer(today) {
        Some(answer) => Json(answer).into_response(),
        None => bad_request("date is out of range"),
    }
}

/// The manifest Helix registers. The bundle hash is computed from the file actually
/// served, so the pin cannot drift from the bytes. Without a built UI it is API-only.
fn manifest(config: &Config) -> Value {
    let base = config.public_url.trim_end_matches('/');
    let mut plugin = json!({
        "name": PLUGIN_NAME,
        "title": "Is it Christmas?",
        "endpoints": [{
            "method": "GET",
            "path": "/v2/workspaces/{workspace}/is-it-christmas",
            "access": "read",
            "upstream_url": format!("{base}/v1/workspaces/{{workspace}}/is-it-christmas"),
        }],
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
            [(header::CONTENT_TYPE, "text/javascript"), (header::CACHE_CONTROL, "no-store")],
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
        .route("/v1/workspaces/{workspace}/is-it-christmas", get(is_it_christmas))
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
        assert_eq!((a.days_until, a.next_christmas.as_str()), (364, "2027-12-25"));
    }

    #[test]
    fn leap_year_days_are_counted() {
        assert_eq!(answer(day(2024, 2, 29)).unwrap().days_until, 300);
    }

    fn temp_bundle(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("is-it-christmas-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
        let res = app.oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    fn app(dir: PathBuf) -> Router {
        router(Config { public_url: "http://127.0.0.1:6010/".into(), bundle_dir: dir })
    }

    #[tokio::test]
    async fn endpoint_uses_the_callers_date() {
        let (status, body) =
            get_json(app(temp_bundle("api")), "/v1/workspaces/default/is-it-christmas?date=2026-12-25").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["is_christmas"], true);
    }

    #[tokio::test]
    async fn endpoint_rejects_a_malformed_date() {
        let (status, body) =
            get_json(app(temp_bundle("bad")), "/v1/workspaces/default/is-it-christmas?date=tomorrow").await;
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
    }

    #[tokio::test]
    async fn manifest_pins_the_bytes_it_serves() {
        let dir = temp_bundle("ui");
        std::fs::write(dir.join("index.js"), b"export const Root = () => null;").unwrap();
        std::fs::write(dir.join("plugin.manifest.json"), br#"{"contractVersion":"1.0.0"}"#).unwrap();
        let (_, body) = get_json(app(dir.clone()), "/plugin.json").await;
        let expected = format!("{:x}", Sha256::digest(b"export const Root = () => null;"));
        assert_eq!(body["plugin"]["bundle"]["sha256"], expected.as_str());
        assert_eq!(body["plugin"]["bundle"]["url"], "http://127.0.0.1:6010/index.js");
        assert_eq!(body["plugin"]["bundle"]["sidecar"]["contractVersion"], "1.0.0");
        assert_eq!(body["plugin"]["navigation"][0]["path"], "/workspaces/{workspace}/plugin/is-it-christmas");
    }
}
