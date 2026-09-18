{{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nemo-platform-zitadel.envoyConfig" -}}
{{- $zitadel := required "nemo-platform.zitadelEnvoy is required" .Values.zitadelEnvoy -}}
{{- $tlsMountPath := "" -}}
{{- range .Values.envoyProxy.extraVolumeMounts -}}
{{- if eq (index . "name") "workload-token-tls" -}}
{{- $tlsMountPath = index . "mountPath" -}}
{{- end -}}
{{- end -}}
{{- $tlsMountPath = required "nemo-platform.envoyProxy.extraVolumeMounts must include workload-token-tls" $tlsMountPath -}}
{{- $apiServiceName := include "nmp-api.api-servicename" . -}}
{{- $publicGateway := required "nemo-platform.zitadelPublicGateway is required" .Values.zitadelPublicGateway -}}
{{- $publicGatewayHost := required "nemo-platform.zitadelPublicGateway.host is required" $publicGateway.host -}}
{{- $publicGatewayPort := required "nemo-platform.zitadelPublicGateway.port is required" $publicGateway.port -}}
{{- $publicGatewayAuthority := printf "%s:%s" $publicGatewayHost (toString $publicGatewayPort) -}}
{{- $spoofHeaders := concat .Values.envoyProxy.trustedHeaders (list "x-nmp-authorized" "x-nmp-scopes") | uniq -}}
admin:
  address:
    socket_address:
      address: 0.0.0.0
      port_value: {{ .Values.envoyProxy.adminPort }}
static_resources:
  listeners:
    - name: listener_0
      address:
        socket_address:
          address: 0.0.0.0
          port_value: {{ .Values.envoyProxy.service.port }}
      filter_chains:
        - transport_socket:
            name: envoy.transport_sockets.tls
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
              common_tls_context:
                tls_certificates:
                  - certificate_chain:
                      filename: {{ printf "%s/tls.crt" $tlsMountPath | quote }}
                    private_key:
                      filename: {{ printf "%s/tls.key" $tlsMountPath | quote }}
          filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: ingress_http
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: nemo
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/.well-known/nemo-platform/"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            path: "/apis/auth/discovery"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            path: "/apis/auth/authenticate"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            prefix: "/apis/auth/ext-authz"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            path: "/apis/auth/jwks"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            path: "/apis/auth/token"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            prefix: "/apis/"
                          route:
                            cluster: nemo
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            path: "/health/gateway/ready"
                          direct_response:
                            status: 503
                            body:
                              inline_string: '{"status":"not_ready"}'
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                        - match:
                            prefix: "/health/"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            path: "/status"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            prefix: "/studio/"
                          route:
                            cluster: nemo
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                          request_headers_to_add:
                            - header:
                                key: x-forwarded-proto
                                value: https
                              append_action: OVERWRITE_IF_EXISTS_OR_ADD
                        - match:
                            prefix: "/"
                          route:
                            cluster: zitadel
                            host_rewrite_literal: {{ $publicGatewayAuthority | quote }}
                          typed_per_filter_config:
                            envoy.filters.http.ext_authz:
                              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute
                              disabled: true
                http_filters:
                  - name: envoy.filters.http.lua
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
                      inline_code: |
                        local function gateway_ready_http_call(request_handle, cluster, authority, path)
                          local ok, response_headers = pcall(function()
                            local headers, _ = request_handle:httpCall(
                              cluster,
                              {
                                [":method"] = "GET",
                                [":path"] = path,
                                [":authority"] = authority
                              },
                              "",
                              5000
                            )
                            return headers
                          end)
                          if not ok then
                            return false, "error"
                          end
                          if response_headers == nil then
                            return false, "missing"
                          end

                          local status = response_headers[":status"] or "missing"
                          return status == "200", status
                        end

                        function envoy_on_request(request_handle)
                          local headers = request_handle:headers()
{{- range $header := $spoofHeaders }}
                          headers:remove({{ $header | quote }})
{{- end }}

                          if headers:get(":path") ~= "/health/gateway/ready" then
                            return
                          end

                          local nemo_ready, nemo_status = gateway_ready_http_call(request_handle, "nemo", {{ $apiServiceName | quote }}, "/health/ready")
                          local zitadel_ready, zitadel_status = gateway_ready_http_call(request_handle, "zitadel", {{ $publicGatewayAuthority | quote }}, "/.well-known/openid-configuration")
                          if nemo_ready and zitadel_ready then
                            request_handle:respond({[":status"] = "200", ["content-type"] = "application/json"}, '{"status":"ready"}')
                            return
                          end

                          request_handle:respond(
                            {[":status"] = "503", ["content-type"] = "application/json"},
                            string.format('{"status":"not_ready","nemo":"%s","zitadel":"%s"}', nemo_status, zitadel_status)
                          )
                        end
                  - name: envoy.filters.http.ext_authz
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthz
                      transport_api_version: V3
                      failure_mode_allow: false
                      status_on_error:
                        code: ServiceUnavailable
                      http_service:
                        server_uri:
                          uri: {{ printf "http://%s:%v" $apiServiceName .Values.api.service.port | quote }}
                          cluster: nemo
                          timeout: 5s
                        path_prefix: "/apis/auth/ext-authz"
                        authorization_response:
                          allowed_upstream_headers:
                            patterns:
                              - exact: x-nmp-principal-id
                              - exact: x-nmp-principal-email
                              - exact: x-nmp-principal-groups
                              - exact: x-nmp-principal-on-behalf-of
                              - exact: x-nmp-principal-on-behalf-of-email
                              - exact: x-nmp-principal-on-behalf-of-groups
                              - exact: x-nmp-scopes
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
  clusters:
    - name: nemo
      connect_timeout: 5s
      type: LOGICAL_DNS
      load_assignment:
        cluster_name: nemo
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: {{ $apiServiceName }}
                      port_value: {{ .Values.api.service.port }}
    - name: zitadel
      connect_timeout: 5s
      type: LOGICAL_DNS
      load_assignment:
        cluster_name: zitadel
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: {{ $zitadel.serviceName }}
                      port_value: {{ $zitadel.servicePort }}
{{- end -}}
