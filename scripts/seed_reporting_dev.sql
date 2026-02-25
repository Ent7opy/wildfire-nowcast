\set ON_ERROR_STOP on

BEGIN;

-- Ensure primary collection exists and capture its id.
WITH existing AS (
    SELECT id
    FROM report_definitions.collections
    WHERE code = 'USA_22'
    ORDER BY id
    LIMIT 1
), inserted AS (
    INSERT INTO report_definitions.collections (
        id,
        name,
        code,
        type,
        owner,
        ref_count,
        created_at,
        updated_at
    )
    SELECT
        900001,
        'USA_22',
        'USA_22',
        'gmaid',
        'seed+usa22@localiq.com',
        0,
        NOW(),
        NOW()
    WHERE NOT EXISTS (SELECT 1 FROM existing)
    RETURNING id
)
SELECT COALESCE((SELECT id FROM existing), (SELECT id FROM inserted)) AS usa22_collection_id
\gset

-- Ensure core refs for the USA_22 collection.
INSERT INTO report_definitions.collection_refs (
    id,
    collection_id,
    ref_id,
    ref_type,
    group_name,
    created_at,
    updated_at
)
SELECT
    900001,
    :usa22_collection_id,
    'USA_22',
    'gmaid',
    'Default',
    NOW(),
    NOW()
WHERE NOT EXISTS (
    SELECT 1
    FROM report_definitions.collection_refs
    WHERE collection_id = :usa22_collection_id
      AND ref_id = 'USA_22'
      AND ref_type = 'gmaid'
);

INSERT INTO report_definitions.collection_refs (
    id,
    collection_id,
    ref_id,
    ref_type,
    group_name,
    created_at,
    updated_at
)
SELECT
    900002,
    :usa22_collection_id,
    'USA22_SEARCH_CORE',
    'gmcid',
    'Default',
    NOW(),
    NOW()
WHERE NOT EXISTS (
    SELECT 1
    FROM report_definitions.collection_refs
    WHERE collection_id = :usa22_collection_id
      AND ref_id = 'USA22_SEARCH_CORE'
      AND ref_type = 'gmcid'
);

INSERT INTO public.collection_metadata (
    id,
    collection_uuid,
    code,
    inserted_at,
    updated_at,
    hidden_tabs,
    collection_id,
    owner,
    gmaid_collection,
    demo_id,
    included_tabs
)
VALUES
    (
        900001,
        'USA_22',
        'USA_22',
        NOW(),
        NOW(),
        ARRAY['debug'],
        :usa22_collection_id,
        'seed+usa22@localiq.com',
        TRUE,
        'demo_usa22',
        ARRAY['overview', 'traffic']
    )
ON CONFLICT (id) DO UPDATE
SET
    collection_uuid = EXCLUDED.collection_uuid,
    code = EXCLUDED.code,
    updated_at = NOW(),
    hidden_tabs = EXCLUDED.hidden_tabs,
    collection_id = EXCLUDED.collection_id,
    owner = EXCLUDED.owner,
    gmaid_collection = EXCLUDED.gmaid_collection,
    demo_id = EXCLUDED.demo_id,
    included_tabs = EXCLUDED.included_tabs;

INSERT INTO public.feature_flags (name, inserted_at, updated_at, global_enabled)
VALUES
    ('seed_ai_summary', NOW(), NOW(), TRUE),
    ('seed_beta_dashboard', NOW(), NOW(), FALSE)
ON CONFLICT (name) DO UPDATE
SET
    updated_at = NOW(),
    global_enabled = EXCLUDED.global_enabled;

INSERT INTO public.feature_flags_on_collections (
    id,
    feature_flag_name,
    collection_metadata_id
)
VALUES
    ('ffc_seed_usa22_1', 'seed_ai_summary', 900001),
    ('ffc_seed_usa22_2', 'seed_beta_dashboard', 900001)
ON CONFLICT (id) DO UPDATE
SET
    feature_flag_name = EXCLUDED.feature_flag_name,
    collection_metadata_id = EXCLUDED.collection_metadata_id;

INSERT INTO public.user_collections (
    id,
    email,
    collection_metadata_id,
    inserted_at,
    updated_at
)
VALUES
    (900001, 'analyst1@example.com', 900001, NOW(), NOW()),
    (900002, 'analyst2@example.com', 900001, NOW(), NOW())
ON CONFLICT (id) DO UPDATE
SET
    email = EXCLUDED.email,
    collection_metadata_id = EXCLUDED.collection_metadata_id,
    updated_at = NOW();

INSERT INTO public.ai_summaries (
    id,
    collection_metadata_id,
    summary,
    inserted_at,
    updated_at,
    messages,
    end_date,
    start_date,
    summaries,
    llm_messages
)
VALUES
    (
        900001,
        900001,
        'USA_22 weekly summary: stable traffic with moderate engagement growth.',
        NOW(),
        NOW(),
        ARRAY[]::jsonb[],
        '2026-02-24',
        '2026-02-17',
        ARRAY[]::jsonb[],
        ARRAY[]::jsonb[]
    )
ON CONFLICT (id) DO UPDATE
SET
    collection_metadata_id = EXCLUDED.collection_metadata_id,
    summary = EXCLUDED.summary,
    updated_at = NOW(),
    end_date = EXCLUDED.end_date,
    start_date = EXCLUDED.start_date;

INSERT INTO public.comments (
    id,
    "reportId",
    entity,
    level,
    title,
    content,
    inserted_at,
    updated_at,
    gmaid,
    collection_metadata_id,
    format,
    user_id
)
VALUES
    (
        'seed_comment_usa22_1',
        'overview',
        'campaign',
        'info',
        'Performance check',
        'USA_22 shows consistent week-over-week growth in engaged sessions.',
        NOW(),
        NOW(),
        'USA_22',
        900001,
        'markdown',
        101
    ),
    (
        'seed_comment_usa22_2',
        'traffic',
        'channel',
        'warning',
        'Organic dip',
        'Organic medium decreased slightly for USA_22 in the latest period.',
        NOW(),
        NOW(),
        'USA_22',
        900001,
        'markdown',
        102
    )
ON CONFLICT (id) DO UPDATE
SET
    "reportId" = EXCLUDED."reportId",
    entity = EXCLUDED.entity,
    level = EXCLUDED.level,
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    updated_at = NOW(),
    gmaid = EXCLUDED.gmaid,
    collection_metadata_id = EXCLUDED.collection_metadata_id,
    format = EXCLUDED.format,
    user_id = EXCLUDED.user_id;

INSERT INTO public.email_feedback (
    id,
    user_email,
    feedback_action,
    feedback_context,
    gmaid,
    inserted_at
)
VALUES
    (900001, 'analyst1@example.com', 'thumbs_up', 'weekly_digest', 'USA_22', NOW()),
    (900002, 'analyst2@example.com', 'thumbs_down', 'ai_summary', 'USA_22', NOW())
ON CONFLICT (id) DO UPDATE
SET
    user_email = EXCLUDED.user_email,
    feedback_action = EXCLUDED.feedback_action,
    feedback_context = EXCLUDED.feedback_context,
    gmaid = EXCLUDED.gmaid;

INSERT INTO public.brand_amplifier (
    id,
    gmaid,
    original_audience,
    lookalike_audience,
    target_list,
    keyword_list,
    inserted_at,
    updated_at,
    gcs_path,
    email_campaign_list,
    xmo_campaign_list
)
VALUES
    (
        900001,
        'USA_22',
        4200,
        7300,
        ARRAY['retargeting_pool_a', 'high_intent_segments'],
        ARRAY['near me', 'open now', 'pricing'],
        NOW(),
        NOW(),
        'gs://seed/reporting_dev/usa_22/brand_amplifier.csv',
        ARRAY['spring_promo', 'always_on'],
        ARRAY['xmo_core_1']
    )
ON CONFLICT (gmaid) DO UPDATE
SET
    original_audience = EXCLUDED.original_audience,
    lookalike_audience = EXCLUDED.lookalike_audience,
    target_list = EXCLUDED.target_list,
    keyword_list = EXCLUDED.keyword_list,
    updated_at = NOW(),
    gcs_path = EXCLUDED.gcs_path,
    email_campaign_list = EXCLUDED.email_campaign_list,
    xmo_campaign_list = EXCLUDED.xmo_campaign_list;

INSERT INTO public.ga_report_data (
    id,
    maid,
    event_date,
    engagement_time_sec,
    city,
    engagement_rate,
    event_count,
    engaged_sessions,
    new_user_count,
    page_view_count,
    page_views_per_session,
    sessions,
    total_users,
    location_name,
    country
)
VALUES
    (900001, 'USA_22', DATE '2026-02-23', 128.4, 'Phoenix', 0.58, 420, 110, 34, 510, 2.2, 232, 265, 'Phoenix, AZ', 'US'),
    (900002, 'USA_22', DATE '2026-02-24', 134.9, 'Tucson', 0.61, 455, 118, 29, 548, 2.3, 238, 272, 'Tucson, AZ', 'US')
ON CONFLICT (id, event_date) DO UPDATE
SET
    maid = EXCLUDED.maid,
    engagement_time_sec = EXCLUDED.engagement_time_sec,
    city = EXCLUDED.city,
    engagement_rate = EXCLUDED.engagement_rate,
    event_count = EXCLUDED.event_count,
    engaged_sessions = EXCLUDED.engaged_sessions,
    new_user_count = EXCLUDED.new_user_count,
    page_view_count = EXCLUDED.page_view_count,
    page_views_per_session = EXCLUDED.page_views_per_session,
    sessions = EXCLUDED.sessions,
    total_users = EXCLUDED.total_users,
    location_name = EXCLUDED.location_name,
    country = EXCLUDED.country;

INSERT INTO public.ga_page_report_data (
    id,
    maid,
    event_date,
    engagement_time_sec,
    engagement_rate,
    event_count,
    engaged_sessions,
    new_user_count,
    page_view_count,
    page_views_per_session,
    sessions,
    total_users,
    base_url,
    page_path,
    page_title,
    traffic_medium,
    channel,
    event_name,
    traffic_source
)
VALUES
    (
        900001,
        'USA_22',
        DATE '2026-02-23',
        77.1,
        0.47,
        196,
        72,
        18,
        250,
        2.0,
        123,
        148,
        'https://example.local',
        '/landing',
        'USA_22 Landing',
        'organic',
        'search',
        'page_view',
        'google'
    ),
    (
        900002,
        'USA_22',
        DATE '2026-02-24',
        83.5,
        0.51,
        212,
        75,
        16,
        278,
        2.1,
        131,
        154,
        'https://example.local',
        '/pricing',
        'USA_22 Pricing',
        'cpc',
        'paid_search',
        'page_view',
        'google_ads'
    )
ON CONFLICT (id, event_date) DO UPDATE
SET
    maid = EXCLUDED.maid,
    engagement_time_sec = EXCLUDED.engagement_time_sec,
    engagement_rate = EXCLUDED.engagement_rate,
    event_count = EXCLUDED.event_count,
    engaged_sessions = EXCLUDED.engaged_sessions,
    new_user_count = EXCLUDED.new_user_count,
    page_view_count = EXCLUDED.page_view_count,
    page_views_per_session = EXCLUDED.page_views_per_session,
    sessions = EXCLUDED.sessions,
    total_users = EXCLUDED.total_users,
    base_url = EXCLUDED.base_url,
    page_path = EXCLUDED.page_path,
    page_title = EXCLUDED.page_title,
    traffic_medium = EXCLUDED.traffic_medium,
    channel = EXCLUDED.channel,
    event_name = EXCLUDED.event_name,
    traffic_source = EXCLUDED.traffic_source;

INSERT INTO public.ga_page_report_data_new (
    id,
    maid,
    event_date,
    engagement_time_sec,
    engagement_rate,
    event_count,
    engaged_sessions,
    new_user_count,
    page_view_count,
    page_views_per_session,
    sessions,
    total_users,
    base_url,
    page_path,
    page_title,
    traffic_medium,
    channel,
    event_name,
    traffic_source
)
VALUES
    (
        900001,
        'USA_22',
        DATE '2026-02-23',
        64.0,
        0.42,
        160,
        58,
        14,
        214,
        1.9,
        112,
        139,
        'https://example.local',
        '/blog/usa22-overview',
        'USA_22 Overview',
        'referral',
        'social',
        'scroll',
        'linkedin'
    ),
    (
        900002,
        'USA_22',
        DATE '2026-02-24',
        70.2,
        0.45,
        175,
        62,
        15,
        228,
        2.0,
        116,
        144,
        'https://example.local',
        '/case-study',
        'USA_22 Case Study',
        'email',
        'email',
        'page_view',
        'newsletter'
    )
ON CONFLICT (id, event_date) DO UPDATE
SET
    maid = EXCLUDED.maid,
    engagement_time_sec = EXCLUDED.engagement_time_sec,
    engagement_rate = EXCLUDED.engagement_rate,
    event_count = EXCLUDED.event_count,
    engaged_sessions = EXCLUDED.engaged_sessions,
    new_user_count = EXCLUDED.new_user_count,
    page_view_count = EXCLUDED.page_view_count,
    page_views_per_session = EXCLUDED.page_views_per_session,
    sessions = EXCLUDED.sessions,
    total_users = EXCLUDED.total_users,
    base_url = EXCLUDED.base_url,
    page_path = EXCLUDED.page_path,
    page_title = EXCLUDED.page_title,
    traffic_medium = EXCLUDED.traffic_medium,
    channel = EXCLUDED.channel,
    event_name = EXCLUDED.event_name,
    traffic_source = EXCLUDED.traffic_source;

INSERT INTO public.nbly_report_data (
    id,
    maid,
    event_date,
    engagement_time_sec,
    engagement_rate,
    event_count,
    engaged_sessions,
    new_user_count,
    page_view_count,
    page_views_per_session,
    sessions,
    total_users,
    traffic_medium,
    event_name,
    traffic_source
)
VALUES
    (900001, 'USA_22', DATE '2026-02-23', 58.9, 0.39, 141, 49, 9, 190, 1.8, 103, 127, 'display', 'click', 'dv360'),
    (900002, 'USA_22', DATE '2026-02-24', 60.7, 0.41, 148, 52, 10, 201, 1.9, 106, 131, 'display', 'click', 'dv360')
ON CONFLICT (id, event_date) DO UPDATE
SET
    maid = EXCLUDED.maid,
    engagement_time_sec = EXCLUDED.engagement_time_sec,
    engagement_rate = EXCLUDED.engagement_rate,
    event_count = EXCLUDED.event_count,
    engaged_sessions = EXCLUDED.engaged_sessions,
    new_user_count = EXCLUDED.new_user_count,
    page_view_count = EXCLUDED.page_view_count,
    page_views_per_session = EXCLUDED.page_views_per_session,
    sessions = EXCLUDED.sessions,
    total_users = EXCLUDED.total_users,
    traffic_medium = EXCLUDED.traffic_medium,
    event_name = EXCLUDED.event_name,
    traffic_source = EXCLUDED.traffic_source;

INSERT INTO public.metadata (
    id,
    filters,
    gmaid,
    user_id,
    report_id,
    uuid,
    resource_id,
    resource_type,
    inserted_at,
    updated_at,
    snapshot,
    notes,
    company_name,
    collection_uuid,
    demo_id
)
VALUES
    (
        900001,
        '{"date_range":{"start":"2026-02-17","end":"2026-02-24"}}'::jsonb,
        'USA_22',
        'seed_user_1',
        'overview',
        'seed-metadata-usa22-1',
        'resource-usa22-1',
        'report',
        NOW(),
        NOW(),
        TRUE,
        ARRAY[]::jsonb[],
        'LocalIQ Seed Co',
        'USA_22',
        'demo_usa22'
    ),
    (
        900002,
        '{"channel":"paid_search"}'::jsonb,
        'USA_22',
        'seed_user_2',
        'traffic',
        'seed-metadata-usa22-2',
        'resource-usa22-2',
        'report',
        NOW(),
        NOW(),
        FALSE,
        ARRAY[]::jsonb[],
        'LocalIQ Seed Co',
        'USA_22',
        'demo_usa22'
    )
ON CONFLICT (id) DO UPDATE
SET
    filters = EXCLUDED.filters,
    gmaid = EXCLUDED.gmaid,
    user_id = EXCLUDED.user_id,
    report_id = EXCLUDED.report_id,
    uuid = EXCLUDED.uuid,
    resource_id = EXCLUDED.resource_id,
    resource_type = EXCLUDED.resource_type,
    updated_at = NOW(),
    snapshot = EXCLUDED.snapshot,
    notes = EXCLUDED.notes,
    company_name = EXCLUDED.company_name,
    collection_uuid = EXCLUDED.collection_uuid,
    demo_id = EXCLUDED.demo_id;

INSERT INTO public.monthly_digest_exports (
    id,
    sent,
    export_id,
    snapshot_id,
    gmaid,
    inserted_at,
    updated_at
)
VALUES
    (900001, TRUE, 'exp_usa22_001', 'snap_usa22_001', 'USA_22', NOW(), NOW()),
    (900002, FALSE, 'exp_usa22_002', 'snap_usa22_002', 'USA_22', NOW(), NOW())
ON CONFLICT (id) DO UPDATE
SET
    sent = EXCLUDED.sent,
    export_id = EXCLUDED.export_id,
    snapshot_id = EXCLUDED.snapshot_id,
    gmaid = EXCLUDED.gmaid,
    updated_at = NOW();

INSERT INTO public.report_cache (
    id,
    snapshot_id,
    report_hash,
    payload,
    args,
    report_type,
    gmaid
)
VALUES
    (
        900001,
        'snap_usa22_001',
        'hash_usa22_overview',
        '{"kpi":"engagement_rate","value":0.61}'::jsonb,
        '{"window":"7d"}'::jsonb,
        'overview',
        'USA_22'
    ),
    (
        900002,
        'snap_usa22_002',
        'hash_usa22_traffic',
        '{"kpi":"sessions","value":238}'::jsonb,
        '{"window":"7d"}'::jsonb,
        'traffic',
        'USA_22'
    )
ON CONFLICT (id) DO UPDATE
SET
    snapshot_id = EXCLUDED.snapshot_id,
    report_hash = EXCLUDED.report_hash,
    payload = EXCLUDED.payload,
    args = EXCLUDED.args,
    report_type = EXCLUDED.report_type,
    gmaid = EXCLUDED.gmaid;

INSERT INTO public.scheduled_exports (
    id,
    gmaid,
    platform_id,
    name,
    frequency,
    report_type,
    report_id,
    filters,
    collection_uuid,
    subject,
    sender,
    message,
    recipients,
    bcc_recipients,
    start_date,
    end_date,
    next_run_date,
    disabled,
    last_sent_at,
    created_at,
    updated_at
)
VALUES
    (
        900001,
        'USA_22',
        1,
        'USA_22 Weekly Overview',
        'weekly'::scheduled_export_frequency,
        'overview',
        'overview_main',
        '{"include_trends":true}'::jsonb,
        'USA_22',
        'USA_22 Weekly Overview',
        'noreply@example.com',
        'Automated weekly export for USA_22.',
        ARRAY['analyst1@example.com', 'analyst2@example.com'],
        ARRAY['manager@example.com'],
        DATE '2026-02-24',
        NULL,
        DATE '2026-03-03',
        FALSE,
        NULL,
        NOW(),
        NOW()
    )
ON CONFLICT (id) DO UPDATE
SET
    gmaid = EXCLUDED.gmaid,
    platform_id = EXCLUDED.platform_id,
    name = EXCLUDED.name,
    frequency = EXCLUDED.frequency,
    report_type = EXCLUDED.report_type,
    report_id = EXCLUDED.report_id,
    filters = EXCLUDED.filters,
    collection_uuid = EXCLUDED.collection_uuid,
    subject = EXCLUDED.subject,
    sender = EXCLUDED.sender,
    message = EXCLUDED.message,
    recipients = EXCLUDED.recipients,
    bcc_recipients = EXCLUDED.bcc_recipients,
    start_date = EXCLUDED.start_date,
    end_date = EXCLUDED.end_date,
    next_run_date = EXCLUDED.next_run_date,
    disabled = EXCLUDED.disabled,
    last_sent_at = EXCLUDED.last_sent_at,
    updated_at = NOW();

INSERT INTO public.scheduled_jobs_metadata (
    id,
    schedule_type,
    schedule_id,
    gmaid,
    metadata,
    inserted_at,
    updated_at
)
VALUES
    (
        900001,
        'scheduled_export',
        '900001',
        'USA_22',
        '{"job":"digest_dispatch","status":"queued"}'::jsonb,
        NOW(),
        NOW()
    ),
    (
        900002,
        'scheduled_export',
        '900001',
        'USA_22',
        '{"job":"cache_refresh","status":"queued"}'::jsonb,
        NOW(),
        NOW()
    )
ON CONFLICT (id) DO UPDATE
SET
    schedule_type = EXCLUDED.schedule_type,
    schedule_id = EXCLUDED.schedule_id,
    gmaid = EXCLUDED.gmaid,
    metadata = EXCLUDED.metadata,
    updated_at = NOW();

COMMIT;
