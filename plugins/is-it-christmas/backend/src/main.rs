// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{env, net::SocketAddr, path::PathBuf};

use is_it_christmas::{router, Config, Override};

#[tokio::main]
async fn main() {
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(6010);
    let public_url = env::var("PUBLIC_URL").unwrap_or_else(|_| format!("http://127.0.0.1:{port}"));
    let bundle_dir = env::var("BUNDLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../dist"));

    // CHRISTMAS_OVERRIDE=yes|no|auto forces the answer for the demo. A typo is an error, not "auto".
    let answer_override: Override = env::var("CHRISTMAS_OVERRIDE")
        .unwrap_or_default()
        .parse()
        .unwrap_or_else(|message: String| {
            eprintln!("{message}");
            std::process::exit(2);
        });

    // Loopback only: Helix reaches this adapter from the same machine in the demo.
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind listener");
    println!("is-it-christmas listening on http://{addr} (answer override: {answer_override:?})");
    println!("register with: nemo plugins install {public_url}/plugin.json");
    axum::serve(
        listener,
        router(Config {
            public_url,
            bundle_dir,
            answer_override,
        }),
    )
    .await
    .expect("serve");
}
