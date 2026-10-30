// Generated from openapi.json. Run npm run generate:api; do not edit.
export interface paths {
    "/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Root */
        get: operations["root__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/agent/stream": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Run Agent Stream Api */
        post: operations["run_agent_stream_api_agent_stream_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/client-config": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Client Config */
        get: operations["get_client_config_client_config_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/download/{filename}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Download File */
        get: operations["download_file_download__filename__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/sessions/{session_id}/answers/{answer_id}/save": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Save Answer */
        post: operations["save_answer_sessions__session_id__answers__answer_id__save_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/sessions/{session_id}/uploads": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Upload Manifest */
        get: operations["get_upload_manifest_sessions__session_id__uploads_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/sessions/{session_id}/uploads/staging": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Stage Browser Upload */
        post: operations["stage_browser_upload_sessions__session_id__uploads_staging_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/sessions/{session_id}/uploads/sync": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Sync Upload Manifest */
        post: operations["sync_upload_manifest_sessions__session_id__uploads_sync_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/slack/default-recipient": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Slack Default Recipient */
        get: operations["get_slack_default_recipient_slack_default_recipient_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** ActionReceipt */
        ActionReceipt: {
            /** @default null */
            artifact?: components["schemas"]["SaveArtifact"] | null;
            /**
             * Error
             * @default null
             */
            error?: string | null;
            /**
             * Error Code
             * @default null
             */
            error_code?: string | null;
            /**
             * File Path
             * @default null
             */
            file_path?: string | null;
            /**
             * Invocation Id
             * @default null
             */
            invocation_id?: string | null;
            /**
             * Kind
             * @enum {string}
             */
            kind: "save_text" | "slack_notify";
            /**
             * Message
             * @default null
             */
            message?: string | null;
            /** @default null */
            operation?: components["schemas"]["SaveOperation"] | null;
            /** @default null */
            slack?: components["schemas"]["SlackDelivery"] | null;
            /**
             * Status
             * @enum {string}
             */
            status: "success" | "error" | "skipped" | "unknown";
            /**
             * Verification
             * @default null
             */
            verification?: ("verified" | "failed" | "unverifiable") | null;
        };
        /** ActionResults */
        ActionResults: {
            /** @default null */
            save_text?: components["schemas"]["SaveTextActionResult"] | null;
            /** @default null */
            slack_notify?: components["schemas"]["SlackDelivery"] | null;
        };
        /** AgentDebugInfo */
        AgentDebugInfo: {
            /** @default null */
            action_results?: components["schemas"]["ActionResults"] | null;
            /** @default null */
            answer_provenance?: components["schemas"]["AnswerProvenance"] | null;
            /** Error Codes */
            error_codes?: ("provider_schema_invalid" | "provider_configuration" | "provider_unavailable" | "provider_rate_limited" | "model_output_invalid" | "model_output_incomplete" | "model_refusal" | "internal_error" | "evidence_insufficient" | "call_budget_exhausted" | "upload_revision_conflict" | "save_failed" | "PLANNER_SCHEMA_INVALID" | "PLANNER_TIMEOUT" | "RETRIEVAL_DOCS_TIMEOUT" | "RETRIEVAL_DOCS_FAILED" | "RAG_INDEX_MISSING" | "LOCAL_RAG_FAILED" | "UPLOAD_RETRIEVER_BUILD_FAILED" | "LLM_STRUCTURED_EMPTY" | "SYNTHESIS_TIMEOUT" | "VALIDATION_UNRESOLVED_REFERENCES" | "VALIDATION_MISSING_CONTENT" | "DEBUG_NORMALIZATION_FAILED" | "SLACK_RECIPIENT_MISSING" | "SLACK_RECIPIENT_INVALID" | "SLACK_RECIPIENT_AMBIGUOUS" | "SLACK_RECIPIENT_CONFLICT" | "SLACK_CONFIGURATION_ERROR" | "SLACK_TARGET_NOT_FOUND" | "SLACK_TARGET_UNAVAILABLE" | "SLACK_PERMISSION_DENIED" | "SLACK_AUTHENTICATION_FAILED" | "SLACK_RATE_LIMITED" | "SLACK_TEMPORARY_FAILURE" | "SLACK_DELIVERY_UNKNOWN" | "SLACK_PROTOCOL_ERROR" | "UPLOAD_PATH_INVALID")[];
            /** Errors */
            errors?: string[];
            /**
             * Execution Evidence
             * @default null
             */
            execution_evidence?: components["schemas"]["ToolExecutionEvidence"] | {
                [key: string]: unknown;
            } | null;
            /** @default null */
            latency_breakdown?: components["schemas"]["LatencyBreakdownModel"] | null;
            /**
             * Latency Ms Server
             * @default null
             */
            latency_ms_server?: number | null;
            /**
             * Llm Calls
             * @default null
             */
            llm_calls?: components["schemas"]["LLMCallRecord"][] | null;
            /** Llm Diagnostics */
            llm_diagnostics?: components["schemas"]["LLMDiagnostic"][];
            /** Memory Compactions */
            memory_compactions?: {
                [key: string]: unknown;
            }[];
            /** Missing Required Debug Fields */
            missing_required_debug_fields?: string[];
            /**
             * Observability Status
             * @enum {string}
             */
            observability_status: "ok" | "degraded" | "failed";
            /** Observed Hits */
            observed_hits?: components["schemas"]["SearchHit"][];
            /** @default null */
            planner_diagnostics?: components["schemas"]["PlannerDiagnostic"] | null;
            /** Planner Errors */
            planner_errors?: string[];
            /** Retrieval Diagnostics */
            retrieval_diagnostics?: components["schemas"]["RetrievalDiagnostic"][];
            /** @default null */
            retry_context?: components["schemas"]["RetryState"] | null;
            /** Route Decisions */
            route_decisions: components["schemas"]["RoutingDecision"][];
            /** Schema Version */
            schema_version: number;
            /**
             * Tool Call Count
             * @default 0
             */
            tool_call_count?: number;
            /** Tool Calls */
            tool_calls?: string[];
            /** Validation Events */
            validation_events?: string[];
        };
        /** AgentRequest */
        AgentRequest: {
            /**
             * Include Debug
             * @default false
             */
            include_debug?: boolean;
            /** Query */
            query: string;
            /** Session Id */
            session_id: string;
            slack_recipient?: components["schemas"]["RecipientInput"] | null;
            uploads: components["schemas"]["UploadContext"];
        };
        /** AgentResponse */
        AgentResponse: {
            /**
             * Answer Id
             * @default null
             */
            answer_id?: string | null;
            /** @default null */
            debug?: components["schemas"]["AgentDebugInfo"] | null;
            /**
             * Message
             * @default
             */
            message?: string;
            /** Missing Slots */
            missing_slots?: string[];
            /** @default null */
            problem?: components["schemas"]["ExecutionProblem"] | null;
            /**
             * Request Id
             * @description ID of this execution, not the domain request ID.
             * @default
             */
            request_id?: string;
            /** @default null */
            response?: components["schemas"]["AnswerResponse"] | null;
            /**
             * Status
             * @default completed
             * @enum {string}
             */
            status?: "completed" | "needs_input" | "failed" | "partial" | "refused";
            /** Trace */
            trace: string;
            /**
             * @description Confirmed attachment snapshot. Null on a failed request means no snapshot was available, not an empty attachment set.
             * @default null
             */
            upload_manifest?: components["schemas"]["UploadManifest"] | null;
        };
        /** AnswerDocument */
        AnswerDocument: {
            /** Blocks */
            blocks?: (components["schemas"]["ParagraphBlock"] | components["schemas"]["ListBlock"] | components["schemas"]["CodeBlock"] | components["schemas"]["TableBlock"] | components["schemas"]["HeadingBlock"])[];
        };
        /**
         * AnswerExportResult
         * @description A saved answer retains its body even when the filesystem operation fails.
         */
        AnswerExportResult: {
            /** Answer Id */
            answer_id: string;
            /** Message */
            message: string;
            /**
             * Request Id
             * @description ID of this export execution, renewed on retry.
             */
            request_id: string;
            response: components["schemas"]["AnswerResponse"];
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "failed";
        };
        /**
         * AnswerProvenance
         * @description The final construction/validation packet and the server-selected source answer.
         *
         *     A packet also exists for deterministic copies and fallbacks without a model
         *     call. It must never be reconstructed by merging a conversation's searches.
         */
        AnswerProvenance: {
            /**
             * Body Kind
             * @enum {string}
             */
            body_kind: "compose" | "acknowledge" | "copy_input" | "transform_input" | "copy_answer" | "transform_answer" | "extract" | "unresolved";
            /**
             * Contract Revision
             * @default null
             */
            contract_revision?: number | null;
            /** Evidence Packet */
            evidence_packet: components["schemas"]["EvidenceRef"][];
            /**
             * Request Id
             * @description Domain request ID from RequestContract, retained across supplementation and retries.
             * @default null
             */
            request_id?: string | null;
            /** Response Hash */
            response_hash: string;
            /**
             * Save Operation Binding Sha256
             * @default null
             */
            save_operation_binding_sha256?: string | null;
            /** @default null */
            source?: components["schemas"]["AnswerSource"] | null;
            /**
             * Version
             * @default 1
             * @constant
             */
            version?: 1;
        };
        /** AnswerResponse */
        AnswerResponse: {
            /** Actions */
            actions?: components["schemas"]["ActionReceipt"][];
            /** Checks */
            checks?: components["schemas"]["UnitCheck"][];
            /** Citations */
            citations?: components["schemas"]["Citation"][];
            content?: components["schemas"]["AnswerDocument"];
            /**
             * Content Hash
             * @default
             */
            content_hash?: string;
            /** Issues */
            issues?: components["schemas"]["ResponseIssue"][];
            /**
             * Retrieval Required
             * @default false
             */
            retrieval_required?: boolean;
            /** @default null */
            semantic_validation?: components["schemas"]["SemanticValidation"] | null;
        };
        /**
         * AnswerSource
         * @description The resolved parent's complete adopted citation set, not its old search candidates.
         */
        AnswerSource: {
            /** Citation Ids */
            citation_ids: string[];
            /**
             * Ref
             * @enum {string}
             */
            ref: "previous" | "pending";
            /** Response Hash */
            response_hash: string;
        };
        /** Body_stage_browser_upload_sessions__session_id__uploads_staging_post */
        Body_stage_browser_upload_sessions__session_id__uploads_staging_post: {
            /** File */
            file: string;
        };
        /** BrowserUploadAddition */
        BrowserUploadAddition: {
            /** Content Hash */
            content_hash: string;
            /** Name */
            name: string;
            /** Replace File Id */
            replace_file_id?: string | null;
            /** Upload Id */
            upload_id: string;
        };
        /** BrowserUploadSyncRequest */
        BrowserUploadSyncRequest: {
            /** Add */
            add?: components["schemas"]["BrowserUploadAddition"][];
            /**
             * Clear
             * @default false
             */
            clear?: boolean;
            /** Epoch */
            epoch: string;
            /** Expected Revision */
            expected_revision: number;
            /** Operation Id */
            operation_id: string;
            /** Remove */
            remove?: string[];
        };
        /** Citation */
        Citation: {
            evidence: components["schemas"]["EvidenceRef"];
            /** Number */
            number: number;
        };
        /**
         * ClientConfig
         * @description Public input limits; server settings and credentials remain private.
         */
        ClientConfig: {
            query: components["schemas"]["ClientQueryConfig"];
            upload: components["schemas"]["ClientUploadConfig"];
        };
        /** ClientQueryConfig */
        ClientQueryConfig: {
            /** Max Chars */
            max_chars: number;
            /** Max Utf8 Bytes */
            max_utf8_bytes: number;
        };
        /** ClientUploadConfig */
        ClientUploadConfig: {
            /** Accepted Suffixes */
            accepted_suffixes: string[];
            /** Max File Bytes */
            max_file_bytes: number;
            /** Max Files */
            max_files: number;
            /** Max Total Bytes */
            max_total_bytes: number;
        };
        /** CodeBlock */
        CodeBlock: {
            content: components["schemas"]["ContentUnit"];
            /**
             * Inline
             * @default false
             */
            inline?: boolean;
            /**
             * Language
             * @default
             */
            language?: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "code";
        };
        /**
         * ContentUnit
         * @description The exact displayed content and its proposed relationship to sources.
         */
        ContentUnit: {
            /**
             * Basis
             * @default interaction
             * @enum {string}
             */
            basis?: "source" | "inference" | "example" | "interaction" | "excerpt";
            /** Refs */
            refs?: string[];
            /** @default null */
            source_location?: components["schemas"]["SourceLocationReference"] | null;
            /** Text */
            text: string;
        };
        /**
         * DocsSearchProviderRequest
         * @description One provider request attempted during this tool invocation.
         */
        DocsSearchProviderRequest: {
            /** Max Results */
            max_results: number;
            /** Query */
            query: string;
            /**
             * Result Count
             * @default 0
             */
            result_count?: number;
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "error";
        };
        /**
         * DocsSearchRequest
         * @description A completed provider scope, reused only within one fixed requirement.
         */
        DocsSearchRequest: {
            /** Max Results */
            max_results: number;
            /** Query */
            query: string;
        };
        /** DocumentElement */
        DocumentElement: {
            /** Anchors */
            anchors?: components["schemas"]["SourceAnchor"][];
            /** Element Id */
            element_id: string;
            /**
             * Heading Level
             * @default null
             */
            heading_level?: number | null;
            /** Heading Path */
            heading_path?: string[];
            /**
             * Kind
             * @enum {string}
             */
            kind: "heading" | "paragraph" | "list" | "code" | "table" | "image";
            /**
             * Language
             * @default null
             */
            language?: string | null;
            /** Metadata */
            metadata?: {
                [key: string]: unknown;
            };
            /**
             * Order
             * @default 0
             */
            order?: number;
            /**
             * Parent Id
             * @default null
             */
            parent_id?: string | null;
            /** @default null */
            table?: components["schemas"]["TableData"] | null;
            /**
             * Text
             * @default
             */
            text?: string;
        };
        /** DocumentSnapshot */
        DocumentSnapshot: {
            /**
             * Capture Scope
             * @default full_document
             * @enum {string}
             */
            capture_scope?: "full_document" | "provider_excerpt";
            /** Content Hash */
            content_hash: string;
            /** Document Id */
            document_id: string;
            /** Media Type */
            media_type: string;
            /** Parser */
            parser: string;
            /** Parser Config */
            parser_config?: {
                [key: string]: unknown;
            };
            /** Parser Version */
            parser_version: string;
            /** Quality Issues */
            quality_issues?: string[];
            /** Snapshot Id */
            snapshot_id: string;
            /**
             * Source Type
             * @enum {string}
             */
            source_type: "official" | "upload";
            /** Source Uri */
            source_uri: string;
            /** Title */
            title: string;
        };
        /** EvidenceRef */
        EvidenceRef: {
            element: components["schemas"]["DocumentElement"];
            /** Id */
            id: string;
            selection?: components["schemas"]["SourceSelection"];
            snapshot: components["schemas"]["DocumentSnapshot"];
        };
        /** ExecutionProblem */
        ExecutionProblem: {
            /**
             * Code
             * @enum {string}
             */
            code: "provider_schema_invalid" | "provider_configuration" | "provider_unavailable" | "provider_rate_limited" | "model_output_invalid" | "model_output_incomplete" | "model_refusal" | "internal_error" | "evidence_insufficient" | "upload_revision_conflict" | "call_budget_exhausted" | "save_failed";
            /** Message */
            message: string;
            /**
             * Next Action
             * @enum {string}
             */
            next_action: "none" | "retry_later" | "fix_configuration" | "supply_information";
            /**
             * Retry After Seconds
             * @default null
             */
            retry_after_seconds?: number | null;
            /** Stage */
            stage: string;
        };
        /** ExplicitRecipient */
        ExplicitRecipient: {
            /**
             * Evidence Ids
             * @default []
             */
            evidence_ids?: string[];
            selector: components["schemas"]["RecipientSelector"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "explicit";
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /** HeadingBlock */
        HeadingBlock: {
            content: components["schemas"]["ContentUnit"];
            /**
             * Level
             * @default 2
             */
            level?: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "heading";
        };
        /** LLMCallRecord */
        LLMCallRecord: {
            /**
             * Attempt
             * @default 0
             */
            attempt?: number;
            /**
             * Endpoint
             * @default null
             */
            endpoint?: string | null;
            /**
             * Failure Type
             * @default null
             */
            failure_type?: string | null;
            /**
             * Latency Ms
             * @default null
             */
            latency_ms?: number | null;
            /**
             * Model Name
             * @default null
             */
            model_name?: string | null;
            /**
             * Path
             * @enum {string}
             */
            path: "direct" | "structured" | "structured_compact_fallback" | "decisions";
            /** Response Metadata */
            response_metadata?: {
                [key: string]: unknown;
            };
            /**
             * Stage
             * @enum {string}
             */
            stage: "summarize" | "planner" | "synthesis" | "pre_synthesis_validation" | "post_synthesis_validation";
            /**
             * Status
             * @default unknown
             * @enum {string}
             */
            status?: "unknown" | "started" | "completed" | "failed";
            usage: components["schemas"]["TokenUsage"];
            /** Usage Metadata */
            usage_metadata?: {
                [key: string]: unknown;
            };
        };
        /**
         * LLMDiagnostic
         * @description Safe metadata only: never store exception text, prompts, or generated values.
         */
        LLMDiagnostic: {
            /**
             * Attempt
             * @default 0
             */
            attempt?: number;
            /**
             * Budget Reason
             * @default null
             */
            budget_reason?: ("calls" | "deadline") | null;
            /**
             * Code
             * @enum {string}
             */
            code: "provider_schema_invalid" | "provider_configuration" | "provider_unavailable" | "provider_rate_limited" | "model_output_invalid" | "model_output_incomplete" | "model_refusal" | "internal_error" | "evidence_insufficient" | "upload_revision_conflict" | "call_budget_exhausted" | "save_failed";
            /**
             * Endpoint
             * @default null
             */
            endpoint?: string | null;
            /**
             * Exception Type
             * @default null
             */
            exception_type?: string | null;
            /**
             * Langchain Openai Version
             * @default null
             */
            langchain_openai_version?: string | null;
            /**
             * Model
             * @default null
             */
            model?: string | null;
            /**
             * Openai Sdk Version
             * @default null
             */
            openai_sdk_version?: string | null;
            /**
             * Provider Code
             * @default null
             */
            provider_code?: string | null;
            /**
             * Provider Param
             * @default null
             */
            provider_param?: string | null;
            /**
             * Provider Request Id
             * @default null
             */
            provider_request_id?: string | null;
            /**
             * Provider Status
             * @default null
             */
            provider_status?: number | null;
            /**
             * Provider Type
             * @default null
             */
            provider_type?: string | null;
            /**
             * Recovery
             * @default stop
             */
            recovery?: string;
            /**
             * Schema Compiler Version
             * @default null
             */
            schema_compiler_version?: string | null;
            /**
             * Schema Hash
             * @default null
             */
            schema_hash?: string | null;
            /**
             * Schema Name
             * @default null
             */
            schema_name?: string | null;
            /** Stage */
            stage: string;
            /**
             * Transport Failure
             * @default null
             */
            transport_failure?: ("timeout" | "connection") | null;
            /** Validation Paths */
            validation_paths?: string[];
        };
        /** LatencyBreakdownModel */
        LatencyBreakdownModel: {
            /**
             * Graph Total Ms
             * @default null
             */
            graph_total_ms?: number | null;
            /** Retrieval Routes */
            retrieval_routes?: components["schemas"]["RetrievalRouteLatencyModel"][];
            /**
             * Server Total Ms
             * @default null
             */
            server_total_ms?: number | null;
            /** Stage Attempts */
            stage_attempts?: components["schemas"]["StageAttemptModel"][];
            stage_totals_ms?: components["schemas"]["StageTotalsModel"];
            /** Synthesis Attempts */
            synthesis_attempts?: components["schemas"]["SynthesisAttemptLatencyModel"][];
            /**
             * Upload Retriever Build Ms
             * @default null
             */
            upload_retriever_build_ms?: number | null;
        };
        /** ListBlock */
        ListBlock: {
            /** Items */
            items: components["schemas"]["ContentUnit"][];
            /**
             * Ordered
             * @default false
             */
            ordered?: boolean;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "list";
        };
        /** OmittedRecipient */
        OmittedRecipient: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "omitted";
        };
        /** ParagraphBlock */
        ParagraphBlock: {
            /** Content */
            content: components["schemas"]["ContentUnit"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "paragraph";
        };
        /**
         * PlannedRequirement
         * @description An independently planned subject has one match mode, never mixed symbol roles.
         */
        PlannedRequirement: {
            /**
             * Aspects
             * @description Literal parameter names, option/flag names, requested value tokens or code fragments that must appear in returned source, such as axis, --force, reflect or random_state. For docs setting-meaning questions, omit ordinary data and arbitrary counts, capacities, sizes, thresholds or seeds; keep their parameter names here and complete settings in query and the body request. A value changing the result is not sufficient to make it a literal anchor. Keep categorical modes, enum members, sentinels and distinct behavior-branch values separately alongside their parameter names, including numeric or Boolean choices. Preserve an explicitly requested exact source/code match. Source locations (lines/ranges, cells, pages or document regions) belong in query and the body request, not aspects, unless explicitly requested as literal strings occurring in source. Classify by the requested role, even when location labels are quoted. Do not repeat the primary API already in symbols. For upload and docs, never put natural-language properties or conditions here; preserve their meaning in query. For docs additionally request the relevant evidence_needs. Only anchors explicitly requested in the user's dialogue belong here; never guess an identifier from the user's description of a property.
             */
            aspects?: string[];
            /**
             * Evidence Needs
             * @description Docs-only evidence kinds requested by the user: signature for call syntax, parameters for option/argument descriptions, examples for usage examples, explanation for conceptual or behavioral explanation. A broad request to explain API usage needs explanation; it does not require all four kinds or examples copied from the documentation. Explaining a named parameter setting requests BOTH parameters and explanation, even when phrased as its behavior or effect; that value needs the parameter's own description. Keep empty for upload. Do not infer unrelated kinds from model knowledge.
             */
            evidence_needs?: ("signature" | "parameters" | "examples" | "explanation")[];
            /**
             * File Ids
             * @description Upload file IDs from the active catalog whose evidence is explicitly requested. Empty searches all active uploads without requiring every file in the answer. For a comparison or inspection of all attachments, include every requested file ID in one task. Empty for docs.
             */
            file_ids?: string[];
            /**
             * Library
             * @description Exact canonical owning-library ID for official docs. Use unsupported for an explicitly requested library outside the enum; preserve its actual name in query/symbols. Python owns standard-library modules such as pathlib. Never append module names, versions or prose to this ID. Null for upload or an unscoped docs subject.
             * @default null
             */
            library?: ("Python" | "Git" | "LangChain" | "matplotlib" | "numpy" | "pandas" | "pytorch" | "Hugging Face" | "fastapi" | "BeautifulSoup" | "streamlit" | "gradio" | "scikit-learn" | "pydantic" | "Bare" | "unsupported") | null;
            /**
             * Match
             * @description topic for broad explanation; symbol for API/usage references, including an uploaded function's implementation together with its actual calls/examples; definition when only the named uploaded function/class implementation is needed. A definition alone excludes calls outside its body.
             * @default topic
             * @enum {string}
             */
            match?: "topic" | "symbol" | "definition";
            /**
             * Symbols
             * @description At most one primary target. Use separate tasks for independent symbols, including each uploaded variable/constant whose value is requested. Never put a second independent variable in the first variable's aspects: symbol retrieval selects only the primary target's occurrences. Calls inspected inside a function belong in its aspects, not additional definitions.
             */
            symbols?: string[];
            /**
             * Version
             * @description One exact requested version token, for example 3.12, 1.26, v2.4 or latest. For 'Python 3.12', return only '3.12'. Never include a library name, spaces, prose or an inferred default version. Null when the user did not constrain the version.
             * @default null
             */
            version?: string | null;
        };
        /** PlannerDiagnostic */
        PlannerDiagnostic: {
            /** Fallback Routes */
            fallback_routes?: string[];
            /**
             * Intent Required
             * @default false
             */
            intent_required?: boolean;
            /**
             * Override Applied
             * @default false
             */
            override_applied?: boolean;
            /**
             * Override Reason
             * @default null
             */
            override_reason?: ("missing_required_retrieval" | "missing_required_routes" | "upload_retriever_missing") | null;
            /** Planner Warnings */
            planner_warnings?: string[];
            /**
             * Reason
             * @default null
             */
            reason?: string | null;
            /** Required Routes */
            required_routes?: string[];
            /**
             * Status
             * @default
             */
            status?: string;
        };
        /**
         * RecipientInput
         * @description An optional form value; incomplete input is meaningful, not omission.
         */
        RecipientInput: {
            /**
             * Kind
             * @enum {string}
             */
            kind: "channel" | "user" | "email";
            /** Value */
            value: string;
        };
        /** RecipientSelection */
        RecipientSelection: {
            /** Request Id */
            request_id: string;
            selector: components["schemas"]["RecipientSelector"];
            /**
             * Source
             * @enum {string}
             */
            source: "user_text" | "request_input" | "configured_default";
        };
        /**
         * RecipientSelector
         * @description A recipient validated when a Slack operation actually selects its target.
         */
        RecipientSelector: {
            /**
             * Kind
             * @enum {string}
             */
            kind: "channel" | "user" | "email";
            /** Value */
            value: string;
        };
        /** ResponseIssue */
        ResponseIssue: {
            /** Code */
            code: string;
            /** Message */
            message: string;
            /**
             * Unit Id
             * @default null
             */
            unit_id?: string | null;
        };
        /** RetrievalDiagnostic */
        RetrievalDiagnostic: {
            /**
             * Answerability
             * @default unknown
             * @enum {string}
             */
            answerability?: "covered" | "partial" | "missing" | "unknown";
            /**
             * Attempt
             * @default 0
             */
            attempt?: number;
            /**
             * Candidate Count
             * @default 0
             */
            candidate_count?: number;
            /** Completed Requests */
            completed_requests?: components["schemas"]["DocsSearchRequest"][];
            /**
             * Error Code
             * @default null
             */
            error_code?: ("provider_schema_invalid" | "provider_configuration" | "provider_unavailable" | "provider_rate_limited" | "model_output_invalid" | "model_output_incomplete" | "model_refusal" | "internal_error" | "evidence_insufficient" | "call_budget_exhausted" | "upload_revision_conflict" | "save_failed" | "PLANNER_SCHEMA_INVALID" | "PLANNER_TIMEOUT" | "RETRIEVAL_DOCS_TIMEOUT" | "RETRIEVAL_DOCS_FAILED" | "RAG_INDEX_MISSING" | "LOCAL_RAG_FAILED" | "UPLOAD_RETRIEVER_BUILD_FAILED" | "LLM_STRUCTURED_EMPTY" | "SYNTHESIS_TIMEOUT" | "VALIDATION_UNRESOLVED_REFERENCES" | "VALIDATION_MISSING_CONTENT" | "DEBUG_NORMALIZATION_FAILED" | "SLACK_RECIPIENT_MISSING" | "SLACK_RECIPIENT_INVALID" | "SLACK_RECIPIENT_AMBIGUOUS" | "SLACK_RECIPIENT_CONFLICT" | "SLACK_CONFIGURATION_ERROR" | "SLACK_TARGET_NOT_FOUND" | "SLACK_TARGET_UNAVAILABLE" | "SLACK_PERMISSION_DENIED" | "SLACK_AUTHENTICATION_FAILED" | "SLACK_RATE_LIMITED" | "SLACK_TEMPORARY_FAILURE" | "SLACK_DELIVERY_UNKNOWN" | "SLACK_PROTOCOL_ERROR" | "UPLOAD_PATH_INVALID") | null;
            /**
             * Evidence Count
             * @default 0
             */
            evidence_count?: number;
            /**
             * Filtered Cross Domain Count
             * @default 0
             */
            filtered_cross_domain_count?: number;
            /**
             * Filtered Invalid Url Count
             * @default 0
             */
            filtered_invalid_url_count?: number;
            /**
             * Filtered Path Prefix Count
             * @default 0
             */
            filtered_path_prefix_count?: number;
            /**
             * Final Evidence Count
             * @default 0
             */
            final_evidence_count?: number;
            /**
             * Include Raw Content Requested
             * @default false
             */
            include_raw_content_requested?: boolean;
            /**
             * Invocation Id
             * @default null
             */
            invocation_id?: string | null;
            /**
             * K
             * @default null
             */
            k?: number | null;
            /**
             * Message
             * @default
             */
            message?: string;
            /**
             * Metric
             * @default
             */
            metric?: string;
            /** Missing Requirements */
            missing_requirements?: string[];
            /**
             * Normalized Score
             * @default null
             */
            normalized_score?: number | null;
            /**
             * Post Filter Ms
             * @default 0
             */
            post_filter_ms?: number;
            /**
             * Provider Ms
             * @default 0
             */
            provider_ms?: number;
            /** Provider Requests */
            provider_requests?: components["schemas"]["DocsSearchProviderRequest"][];
            /**
             * Provider Result Count
             * @default 0
             */
            provider_result_count?: number;
            /**
             * Query
             * @default
             */
            query?: string;
            /**
             * Raw Score
             * @default null
             */
            raw_score?: number | null;
            /**
             * Request Fingerprint
             * @default
             */
            request_fingerprint?: string;
            /**
             * Requirement Id
             * @default
             */
            requirement_id?: string;
            /**
             * Result Count
             * @default 0
             */
            result_count?: number;
            /**
             * Reused
             * @default false
             */
            reused?: boolean;
            /**
             * Route
             * @default
             */
            route?: string;
            /**
             * Score Direction
             * @default
             * @enum {string}
             */
            score_direction?: "higher_is_better" | "lower_is_better" | "";
            /**
             * Status
             * @default
             */
            status?: string;
            /**
             * Tool
             * @default
             */
            tool?: string;
            /** Warnings */
            warnings?: string[];
        };
        /** RetrievalRouteLatencyModel */
        RetrievalRouteLatencyModel: {
            /**
             * Attempt
             * @default 1
             */
            attempt?: number;
            /**
             * Latency Ms
             * @default 0
             */
            latency_ms?: number;
            /**
             * Route
             * @default
             */
            route?: string;
            /**
             * Status
             * @default
             */
            status?: string;
            /**
             * Tool
             * @default
             */
            tool?: string;
        };
        /** RetrievalScore */
        RetrievalScore: {
            /**
             * Direction
             * @enum {string}
             */
            direction: "higher" | "lower";
            /** Metric */
            metric: string;
            /**
             * Normalized
             * @default null
             */
            normalized?: number | null;
            /**
             * Raw
             * @default null
             */
            raw?: number | null;
        };
        /** RetrievalTask */
        RetrievalTask: {
            /** K */
            k: number;
            /** Query */
            query: string;
            requirement?: components["schemas"]["PlannedRequirement"];
            /**
             * Requirement Id
             * @description A stable short ID for this independent requirement; preserve it during retries.
             * @default
             */
            requirement_id?: string;
            /**
             * Route
             * @enum {string}
             */
            route: "docs" | "upload";
        };
        /** RetryState */
        RetryState: {
            /**
             * Attempt
             * @default 0
             */
            attempt?: number;
            /**
             * Content Repair Attempt
             * @default 0
             */
            content_repair_attempt?: number;
            /** Failed Requirement Ids */
            failed_requirement_ids?: string[];
            /** Failed Routes */
            failed_routes?: string[];
            /**
             * Hit Start Index
             * @default 0
             */
            hit_start_index?: number;
            /**
             * Max Content Repairs
             * @default 1
             */
            max_content_repairs?: number;
            /**
             * Max Retries
             * @default 1
             */
            max_retries?: number;
            /**
             * Needs Retry
             * @default false
             */
            needs_retry?: boolean;
            /** Original Tasks */
            original_tasks?: components["schemas"]["RetrievalTask"][];
            /** Preserved Hits */
            preserved_hits?: components["schemas"]["SearchHit"][];
            /** Preserved Retrieval Diagnostics */
            preserved_retrieval_diagnostics?: components["schemas"]["RetrievalDiagnostic"][];
            /**
             * Retrieval Diagnostic Start Index
             * @default 0
             */
            retrieval_diagnostic_start_index?: number;
            /**
             * Retrieval Error Start Index
             * @default 0
             */
            retrieval_error_start_index?: number;
            /**
             * Retrieval Feedback
             * @default
             */
            retrieval_feedback?: string;
            /**
             * Retry Reason
             * @default null
             */
            retry_reason?: ("no_evidence" | "low_score" | "tool_error" | "blocked_missing_upload" | "unresolved_references" | "missing" | "missing_route_coverage" | "missing_content") | null;
            /**
             * Retry Scope
             * @default refresh_routes
             * @enum {string}
             */
            retry_scope?: "refresh_routes" | "reuse_hits_resynthesize";
            /**
             * Score Avg
             * @default null
             */
            score_avg?: number | null;
            /** Terminal Failed Requirement Ids */
            terminal_failed_requirement_ids?: string[];
            /**
             * Terminal Retrieval Reason
             * @default null
             */
            terminal_retrieval_reason?: ("no_evidence" | "low_score" | "tool_error" | "blocked_missing_upload" | "unresolved_references" | "missing" | "missing_route_coverage" | "missing_content") | null;
        };
        /**
         * RoutingDecision
         * @description One committed choice of the next node, shared by execution and diagnostics.
         */
        RoutingDecision: {
            /** Reason */
            reason: string;
            /** Sequence */
            sequence: number;
            /**
             * Source
             * @enum {string}
             */
            source: "add_user_message" | "planner" | "pre_synthesis_validation" | "post_synthesis_validation";
            /**
             * Target
             * @enum {string}
             */
            target: "summarize_old_messages" | "planner" | "retrieve_dispatch" | "pre_synthesis_validation" | "synthesize" | "action_postprocess";
        };
        /** SaveArtifact */
        SaveArtifact: {
            /** Artifact Id */
            artifact_id: string;
            /** Byte Count */
            byte_count: number;
            /** Created At */
            created_at: number;
            /** Expires At */
            expires_at: number;
            /** Filename */
            filename: string;
            /** Sha256 */
            sha256: string;
        };
        /** SaveOperation */
        SaveOperation: {
            /** Answer Hash */
            answer_hash: string;
            /** Byte Count */
            byte_count: number;
            /** Contract Revision */
            contract_revision: number;
            /**
             * Export Profile
             * @default answer-text-v1/utf-8-sig
             * @constant
             */
            export_profile?: "answer-text-v1/utf-8-sig";
            /** Operation Id */
            operation_id: string;
            /** Payload Sha256 */
            payload_sha256: string;
            /** Request Id */
            request_id: string;
            /** Session Id */
            session_id: string;
            /** Source Hash */
            source_hash: string;
            /**
             * Target Kind
             * @enum {string}
             */
            target_kind: "compose" | "copy_input" | "transform_input" | "copy_answer" | "transform_answer" | "extract";
        };
        /** SaveTextActionResult */
        SaveTextActionResult: {
            /**
             * Bytes
             * @default 0
             */
            bytes?: number;
            /**
             * Error
             * @default null
             */
            error?: string | null;
            /**
             * Error Code
             * @default null
             */
            error_code?: ("provider_schema_invalid" | "provider_configuration" | "provider_unavailable" | "provider_rate_limited" | "model_output_invalid" | "model_output_incomplete" | "model_refusal" | "internal_error" | "evidence_insufficient" | "call_budget_exhausted" | "upload_revision_conflict" | "save_failed" | "PLANNER_SCHEMA_INVALID" | "PLANNER_TIMEOUT" | "RETRIEVAL_DOCS_TIMEOUT" | "RETRIEVAL_DOCS_FAILED" | "RAG_INDEX_MISSING" | "LOCAL_RAG_FAILED" | "UPLOAD_RETRIEVER_BUILD_FAILED" | "LLM_STRUCTURED_EMPTY" | "SYNTHESIS_TIMEOUT" | "VALIDATION_UNRESOLVED_REFERENCES" | "VALIDATION_MISSING_CONTENT" | "DEBUG_NORMALIZATION_FAILED" | "SLACK_RECIPIENT_MISSING" | "SLACK_RECIPIENT_INVALID" | "SLACK_RECIPIENT_AMBIGUOUS" | "SLACK_RECIPIENT_CONFLICT" | "SLACK_CONFIGURATION_ERROR" | "SLACK_TARGET_NOT_FOUND" | "SLACK_TARGET_UNAVAILABLE" | "SLACK_PERMISSION_DENIED" | "SLACK_AUTHENTICATION_FAILED" | "SLACK_RATE_LIMITED" | "SLACK_TEMPORARY_FAILURE" | "SLACK_DELIVERY_UNKNOWN" | "SLACK_PROTOCOL_ERROR" | "UPLOAD_PATH_INVALID") | null;
            /**
             * File Path
             * @default null
             */
            file_path?: string | null;
            /**
             * Message
             * @default null
             */
            message?: string | null;
            /**
             * Status
             * @default
             */
            status?: string;
        };
        /** SearchHit */
        SearchHit: {
            evidence: components["schemas"]["EvidenceRef"];
            /** Rank */
            rank: number;
            /**
             * Requirement Id
             * @default
             */
            requirement_id?: string;
            score: components["schemas"]["RetrievalScore"];
        };
        /**
         * SemanticCheck
         * @description One evaluated probability or explicit refusal from the Decisions API.
         */
        SemanticCheck: {
            /** Name */
            name: string;
            /** Probability */
            probability: number | null;
            /**
             * Status
             * @default evaluated
             * @enum {string}
             */
            status?: "evaluated" | "refused";
            /**
             * Unit Id
             * @default null
             */
            unit_id?: string | null;
        };
        /**
         * SemanticValidation
         * @description Semantic results bound to the exact answer and cited source revisions.
         */
        SemanticValidation: {
            /** Checks */
            checks: components["schemas"]["SemanticCheck"][];
            /** Content Hash */
            content_hash: string;
            /** Evidence Fingerprint */
            evidence_fingerprint: string;
            /**
             * Model
             * @default gpt-6-luna
             */
            model?: string;
            /** Threshold */
            threshold: number;
        };
        /** SlackDefault */
        SlackDefault: {
            failure?: components["schemas"]["SlackFailure"] | null;
            selector?: components["schemas"]["RecipientSelector"] | null;
        };
        /**
         * SlackDelivery
         * @description The selected identity survives partial resolution and later retries.
         *
         *     Optional IDs mean resolution has not reached that stage. Failure is always
         *     explicit in status/failure, never represented by an absent ID alone.
         */
        SlackDelivery: {
            /** @default null */
            failure?: components["schemas"]["SlackFailure"] | null;
            /** Intent */
            intent: components["schemas"]["OmittedRecipient"] | components["schemas"]["ExplicitRecipient"] | components["schemas"]["UnresolvedRecipient"];
            /**
             * Invocation Id
             * @default null
             */
            invocation_id?: string | null;
            /**
             * Message Ts
             * @default null
             */
            message_ts?: string | null;
            /**
             * Resolved User Id
             * @default null
             */
            resolved_user_id?: string | null;
            /** @default null */
            selection?: components["schemas"]["RecipientSelection"] | null;
            /**
             * Status
             * @default pending
             * @enum {string}
             */
            status?: "pending" | "sent" | "not_sent" | "unknown";
            /** @default null */
            target?: components["schemas"]["SlackTarget"] | null;
        };
        /** SlackFailure */
        SlackFailure: {
            /**
             * Code
             * @enum {string}
             */
            code: "recipient_missing" | "recipient_invalid" | "recipient_ambiguous" | "recipient_conflict" | "configuration_error" | "target_not_found" | "target_unavailable" | "permission_denied" | "authentication_failed" | "rate_limited" | "temporary_failure" | "delivery_unknown" | "protocol_error";
            /** Message */
            message: string;
            /**
             * Next Action
             * @enum {string}
             */
            next_action: "correct_input" | "fix_configuration" | "retry_same_target" | "verify_delivery";
            /**
             * Retry After Seconds
             * @default null
             */
            retry_after_seconds?: number | null;
            /**
             * Slack Error
             * @default null
             */
            slack_error?: string | null;
            /**
             * Stage
             * @enum {string}
             */
            stage: "input" | "selection" | "lookup" | "open_dm" | "send";
        };
        /** SlackTarget */
        SlackTarget: {
            /** Channel Id */
            channel_id: string;
            /**
             * User Id
             * @default null
             */
            user_id?: string | null;
        };
        /**
         * SourceAnchor
         * @description Known source location; absent coordinates are valid, never invented.
         *
         *     ``start``/``end`` are a half-open range in the adapter's original text.
         *     Lines/pages are one-based; notebook cell indexes are zero-based. Bounding
         *     boxes use ordered (left, minimum_y, right, maximum_y) coordinates.
         */
        SourceAnchor: {
            /**
             * Bbox
             * @default null
             */
            bbox?: [
                number,
                number,
                number,
                number
            ] | null;
            /**
             * Cell Id
             * @default null
             */
            cell_id?: string | null;
            /**
             * Cell Index
             * @default null
             */
            cell_index?: number | null;
            /**
             * Coordinate Origin
             * @default top_left
             * @enum {string}
             */
            coordinate_origin?: "top_left" | "bottom_left";
            /**
             * Coordinate Space
             * @default normalized
             * @enum {string}
             */
            coordinate_space?: "normalized" | "points" | "pixels";
            /**
             * End
             * @default null
             */
            end?: number | null;
            /**
             * Kind
             * @enum {string}
             */
            kind: "text" | "code" | "notebook" | "web" | "page" | "table";
            /**
             * Line End
             * @default null
             */
            line_end?: number | null;
            /**
             * Line Start
             * @default null
             */
            line_start?: number | null;
            /**
             * Page Height
             * @default null
             */
            page_height?: number | null;
            /**
             * Page No
             * @default null
             */
            page_no?: number | null;
            /**
             * Page Width
             * @default null
             */
            page_width?: number | null;
            /**
             * Precision
             * @default element
             * @enum {string}
             */
            precision?: "exact" | "element" | "page" | "document";
            /**
             * Start
             * @default null
             */
            start?: number | null;
        };
        /**
         * SourceLocationReference
         * @description A nonempty element-character range inside one supplied evidence selection.
         */
        SourceLocationReference: {
            /** End */
            end: number;
            /** Ref */
            ref: string;
            /** Start */
            start: number;
        };
        /**
         * SourceSelection
         * @description Half-open offsets in element.text, or structured table cell IDs.
         */
        SourceSelection: {
            /** Cell Ids */
            cell_ids?: string[];
            /**
             * End
             * @default null
             */
            end?: number | null;
            /**
             * Start
             * @default 0
             */
            start?: number;
        };
        /** StageAttemptModel */
        StageAttemptModel: {
            /**
             * Attempt
             * @default 1
             */
            attempt?: number;
            /**
             * Latency Ms
             * @default 0
             */
            latency_ms?: number;
            /**
             * Stage
             * @enum {string}
             */
            stage: "summarize" | "planner" | "retrieval" | "pre_synthesis_validation" | "synthesis" | "post_synthesis_validation" | "validation" | "action_postprocess";
            /**
             * Status
             * @default null
             */
            status?: string | null;
        };
        /** StageTotalsModel */
        StageTotalsModel: {
            /**
             * Action Postprocess Ms
             * @default 0
             */
            action_postprocess_ms?: number;
            /**
             * Planner Ms
             * @default 0
             */
            planner_ms?: number;
            /**
             * Post Synthesis Validation Ms
             * @default 0
             */
            post_synthesis_validation_ms?: number;
            /**
             * Pre Synthesis Validation Ms
             * @default 0
             */
            pre_synthesis_validation_ms?: number;
            /**
             * Retrieval Total Ms
             * @default 0
             */
            retrieval_total_ms?: number;
            /**
             * Summarize Ms
             * @default 0
             */
            summarize_ms?: number;
            /**
             * Synthesis Total Ms
             * @default 0
             */
            synthesis_total_ms?: number;
            /**
             * Validation Ms
             * @default 0
             */
            validation_ms?: number;
        };
        /**
         * StagedUpload
         * @description A received file that has not yet changed the searchable attachment set.
         */
        StagedUpload: {
            /** Content Hash */
            content_hash: string;
            /** Name */
            name: string;
            /** Size Bytes */
            size_bytes: number;
            /** Upload Id */
            upload_id: string;
        };
        /** SynthesisAttemptLatencyModel */
        SynthesisAttemptLatencyModel: {
            /**
             * Attempt
             * @default 1
             */
            attempt?: number;
            /**
             * Fallback Ms
             * @default null
             */
            fallback_ms?: number | null;
            /**
             * Mode
             * @enum {string}
             */
            mode: "structured_only" | "structured_empty_fallback" | "timeout_grounded_fallback" | "structured_error_plain_fallback" | "compact_structured_fallback" | "plain_summary_attach_fallback" | "korean_template_summary_fallback" | "deterministic_grounded_fallback" | "deterministic_grounded_direct";
            /**
             * Structured Ms
             * @default null
             */
            structured_ms?: number | null;
            /**
             * Total Ms
             * @default 0
             */
            total_ms?: number;
        };
        /** TableBlock */
        TableBlock: {
            /** Columns */
            columns: components["schemas"]["ContentUnit"][];
            /** Rows */
            rows?: components["schemas"]["ContentUnit"][][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "table";
        };
        /** TableCell */
        TableCell: {
            /** Anchors */
            anchors?: components["schemas"]["SourceAnchor"][];
            /** Cell Id */
            cell_id: string;
            /** Col */
            col: number;
            /**
             * Col Span
             * @default 1
             */
            col_span?: number;
            /**
             * Is Header
             * @default false
             */
            is_header?: boolean;
            /** Row */
            row: number;
            /**
             * Row Span
             * @default 1
             */
            row_span?: number;
            /** Text */
            text: string;
        };
        /** TableData */
        TableData: {
            /** Cells */
            cells?: components["schemas"]["TableCell"][];
        };
        /** TokenUsage */
        TokenUsage: {
            /**
             * Cache Write Input Tokens
             * @default null
             */
            cache_write_input_tokens?: number | null;
            /**
             * Cached Input Tokens
             * @default null
             */
            cached_input_tokens?: number | null;
            /**
             * Input Tokens
             * @default null
             */
            input_tokens?: number | null;
            /** Issues */
            issues?: components["schemas"]["UsageIssue"][];
            /**
             * Output Tokens
             * @default null
             */
            output_tokens?: number | null;
            /**
             * Reasoning Output Tokens
             * @default null
             */
            reasoning_output_tokens?: number | null;
        };
        /** ToolExecutionEvent */
        ToolExecutionEvent: {
            /** Invocation Id */
            invocation_id: string;
            /**
             * Origin Invocation Id
             * @default null
             */
            origin_invocation_id?: string | null;
            /**
             * Phase
             * @enum {string}
             */
            phase: "started" | "succeeded" | "failed" | "blocked" | "reused";
            /**
             * Reason Code
             * @default null
             */
            reason_code?: string | null;
            /** Sequence */
            sequence: number;
            /** Tool Name */
            tool_name: string;
        };
        /** ToolExecutionEvidence */
        ToolExecutionEvidence: {
            /** Events */
            events: components["schemas"]["ToolExecutionEvent"][];
            /**
             * Request Id
             * @description ID of this execution, matching TurnResult.request_id; not the domain request ID.
             */
            request_id: string;
            /**
             * Schema Version
             * @constant
             */
            schema_version: 1;
            /**
             * Status
             * @enum {string}
             */
            status: "complete" | "incomplete" | "unavailable";
        };
        /** UnitCheck */
        UnitCheck: {
            /** Issues */
            issues?: string[];
            /**
             * Reference Status
             * @enum {string}
             */
            reference_status: "resolved" | "missing" | "not_required";
            /**
             * Support Status
             * @default not_evaluated
             * @enum {string}
             */
            support_status?: "not_evaluated" | "exact_match" | "unsupported";
            /** Unit Id */
            unit_id: string;
        };
        /** UnresolvedRecipient */
        UnresolvedRecipient: {
            /**
             * Evidence Ids
             * @default []
             */
            evidence_ids?: string[];
            /** Raw Input */
            raw_input: string;
            /**
             * Reason
             * @enum {string}
             */
            reason: "invalid" | "ambiguous" | "unverified" | "conflict";
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            state: "unresolved";
        };
        /** UploadAddition */
        UploadAddition: {
            /** Content Hash */
            content_hash: string;
            /** Name */
            name: string;
            /** Path */
            path: string;
            /** Replace File Id */
            replace_file_id?: string | null;
        };
        /** UploadContext */
        UploadContext: {
            /** Epoch */
            epoch: string;
            /** Revision */
            revision: number;
        };
        /** UploadFileInfo */
        UploadFileInfo: {
            /** Content Hash */
            content_hash: string;
            /** File Id */
            file_id: string;
            /** Name */
            name: string;
            /** Size Bytes */
            size_bytes: number;
            /** Source Uri */
            source_uri: string;
        };
        /** UploadManifest */
        UploadManifest: {
            /** Epoch */
            epoch: string;
            /** Files */
            files: components["schemas"]["UploadFileInfo"][];
            /** Revision */
            revision: number;
        };
        /** UploadSyncRequest */
        UploadSyncRequest: {
            /** Add */
            add?: components["schemas"]["UploadAddition"][];
            /**
             * Clear
             * @default false
             */
            clear?: boolean;
            /** Epoch */
            epoch: string;
            /** Expected Revision */
            expected_revision: number;
            /** Operation Id */
            operation_id: string;
            /** Remove */
            remove?: string[];
        };
        /** UsageIssue */
        UsageIssue: {
            /**
             * Code
             * @enum {string}
             */
            code: "invalid_value" | "conflict" | "total_mismatch" | "detail_exceeds_total";
            /**
             * Field
             * @enum {string}
             */
            field: "input_tokens" | "output_tokens" | "total_tokens" | "cached_input_tokens" | "cache_write_input_tokens" | "reasoning_output_tokens" | "metadata";
            /** Source */
            source: string;
        };
        /** ValidationError */
        ValidationError: {
            /** Context */
            ctx?: Record<string, never>;
            /** Input */
            input?: unknown;
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
        AgentStreamEvent_done: Record<string, never>;
        AgentStreamEvent_error: {
            message: string;
            /** @enum {string} */
            stage?: "summarize" | "planner" | "retrieval" | "pre_synthesis_validation" | "synthesis" | "post_synthesis_validation" | "validation" | "action_postprocess";
        };
        /** AgentResponse */
        AgentStreamEvent_final_response: {
            /**
             * Answer Id
             * @default null
             */
            answer_id?: string | null;
            /** @default null */
            debug?: components["schemas"]["AgentDebugInfo"] | null;
            /**
             * Message
             * @default
             */
            message?: string;
            /** Missing Slots */
            missing_slots?: string[];
            /** @default null */
            problem?: components["schemas"]["ExecutionProblem"] | null;
            /**
             * Request Id
             * @description ID of this execution, not the domain request ID.
             * @default
             */
            request_id?: string;
            /** @default null */
            response?: components["schemas"]["AnswerResponse"] | null;
            /**
             * Status
             * @default completed
             * @enum {string}
             */
            status?: "completed" | "needs_input" | "failed" | "partial" | "refused";
            /** Trace */
            trace: string;
            /**
             * @description Confirmed attachment snapshot. Null on a failed request means no snapshot was available, not an empty attachment set.
             * @default null
             */
            upload_manifest?: components["schemas"]["UploadManifest"] | null;
        };
        AgentStreamEvent_heartbeat: {
            attempt: number;
            elapsed_ms: number;
            /** @enum {string} */
            stage: "summarize" | "planner" | "retrieval" | "pre_synthesis_validation" | "synthesis" | "post_synthesis_validation" | "validation" | "action_postprocess";
        };
        AgentStreamEvent_progress_snapshot: {
            /** @enum {string} */
            stage: "summarize" | "planner" | "retrieval" | "pre_synthesis_validation" | "synthesis" | "post_synthesis_validation" | "validation" | "action_postprocess";
            summary: string;
        };
        AgentStreamEvent_request_started: {
            request_id: string;
            session_id: string;
        };
        AgentStreamEvent_stage_completed: {
            attempt: number;
            latency_ms?: number;
            /** @enum {string} */
            stage: "summarize" | "planner" | "retrieval" | "pre_synthesis_validation" | "synthesis" | "post_synthesis_validation" | "validation" | "action_postprocess";
            status?: string;
        };
        AgentStreamEvent_stage_started: {
            attempt: number;
            /** @enum {string} */
            stage: "summarize" | "planner" | "retrieval" | "pre_synthesis_validation" | "synthesis" | "post_synthesis_validation" | "validation" | "action_postprocess";
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    root__get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    run_agent_stream_api_agent_stream_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AgentRequest"];
            };
        };
        responses: {
            /** @description UTF-8 SSE frames: event: <name>\ndata: <JSON object>\n\n. Each execution ends with one final_response carrying status, response, problem, message, request_id, trace, debug and upload_manifest. Status distinguishes completed, needs_input, failed, partial and refused; a failed result has no answer document. The manifest is captured under the session lock; null on failure means unknown state, not an empty attachment set. HTTP 200 and done alone do not indicate success. Clients must not automatically resubmit interrupted requests because agent actions may already have executed. x-sse-events maps each event name to its data schema. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    /**
                     * @example event: request_started
                     *     data: {"request_id":"abc12345","session_id":"demo"}
                     *
                     *     event: final_response
                     *     data: {"status":"failed","response":null,"message":"","problem":{"code":"internal_error","stage":"request","message":"요청을 완료하지 못했습니다.","next_action":"none"},"request_id":"abc12345","missing_slots":[],"trace":"Request ID: abc12345","debug":null,"upload_manifest":null}
                     *
                     *     event: done
                     *     data: {}
                     */
                    "text/event-stream": string;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_client_config_client_config_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ClientConfig"];
                };
            };
        };
    };
    download_file_download__filename__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                filename: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    save_answer_sessions__session_id__answers__answer_id__save_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                session_id: string;
                answer_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AnswerExportResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_upload_manifest_sessions__session_id__uploads_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                session_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UploadManifest"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    stage_browser_upload_sessions__session_id__uploads_staging_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                session_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_stage_browser_upload_sessions__session_id__uploads_staging_post"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StagedUpload"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    sync_upload_manifest_sessions__session_id__uploads_sync_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                session_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UploadSyncRequest"] | components["schemas"]["BrowserUploadSyncRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UploadManifest"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_slack_default_recipient_slack_default_recipient_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SlackDefault"];
                };
            };
        };
    };
}

export interface AgentStreamEventData {
    "done": components["schemas"]["AgentStreamEvent_done"];
    "error": components["schemas"]["AgentStreamEvent_error"];
    "final_response": components["schemas"]["AgentStreamEvent_final_response"];
    "heartbeat": components["schemas"]["AgentStreamEvent_heartbeat"];
    "progress_snapshot": components["schemas"]["AgentStreamEvent_progress_snapshot"];
    "request_started": components["schemas"]["AgentStreamEvent_request_started"];
    "stage_completed": components["schemas"]["AgentStreamEvent_stage_completed"];
    "stage_started": components["schemas"]["AgentStreamEvent_stage_started"];
}
