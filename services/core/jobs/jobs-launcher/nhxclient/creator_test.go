// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSecretEffectiveCreator(t *testing.T) {
	for _, delegated := range []bool{false, true} {
		t.Run(map[bool]string{false: "direct", true: "delegated"}[delegated], func(t *testing.T) {
			creator := `{"id":"creator","email":"creator@example.test","groups":["readers"],"account_id":"account-creator","authz_aliases":["creator-alias"]}`
			if delegated {
				creator = `{"id":"service:plugin","groups":["actor-group"],"on_behalf_of":"creator","on_behalf_of_email":"creator@example.test","on_behalf_of_groups":["readers"],"on_behalf_of_account_id":"account-creator","on_behalf_of_authz_aliases":["creator-alias"]}`
			}
			var principal Principal
			if err := json.Unmarshal([]byte(creator), &principal); err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				expected := map[string]string{"X-NHX-Principal-Id": "service:jobs", "X-NHX-Principal-On-Behalf-Of": "creator", "X-NHX-Principal-On-Behalf-Of-Email": "creator@example.test", "X-NHX-Principal-On-Behalf-Of-Groups": "readers", "X-NHX-Subject-Account-Id": "account-creator", "X-NHX-Subject-Aliases": "creator-alias"}
				for k, v := range expected {
					if r.Header.Get(k) != v {
						t.Errorf("%s: got %q, want %q", k, r.Header.Get(k), v)
					}
				}
				if r.Header.Get("X-NHX-Principal-Groups") != "" {
					t.Error("creator groups leaked to actor")
				}
				w.Write([]byte(`{"value":"synthetic"}`))
			}))
			defer server.Close()

			if _, err := NewSecretClient(server.URL, &principal).GetSecret(context.Background(), "ws", "secret"); err != nil {
				t.Fatal(err)
			}
		})
	}
}
