OUTPUTS OF THIS RUN
regions_map.png        embedding map; left = significant regions (label #rank), right = anomalous
                       flows coloured by temporal pattern
clustering.png         KL significant regions (left) vs filter-marked flows (right)
sweep_sigalpha.png     this run's regions at each threshold of --sig-alphas (one panel per
                       threshold; a region keeps its colour); counts in sweep_sigalpha_counts.csv
anomalous_flows.csv    one row per flow in a significant region (p <= sig-alpha):
  region_rank/region_id/region_p_value/region_share_of_divergence/region_service
  conn_id, protocol    unit id ("ip:port<->ip:port"; host runs: the IP; router logs:
                       "router|device|proto/port") and L4 protocol
  router, device       router folder and device (MAC as logged) -- router-log data only
  oui, is_iot          device vendor prefix (first 3 MAC bytes) and IoT flag (new router logs)
  conn_direction       local / out / in (new router logs)
  has_ech              1 if the connection had a client hello with a mitm_force SNI (ECH)
  src_ip/src_port, dst_ip/dst_port   dst = endpoint with the LOWER port (service side);
                       router logs: src = device, dst_port from the log, no IPs
  service              name of the destination port (HTTPS/QUIC, DNS, ...; hi:<port> if unknown)
  start_time/end_time/duration_h/span_bins/active_bins/active_frac    timing of the flow
  temporal_pattern     continuous | single-burst | periodic (~N min) | sporadic | intermittent
  n_bursts/max_burst_bins/median_gap_bins/gap_cv   burst structure (gap_cv ~0 = regular)
  period_min/period_strength   dominant period of activity (autocorrelation peak, 0..1)
  peak_time/peak_packets/dominant_hour
  <channel>_mean_active/<channel>_max   for channels ['packet_count', 'fwd_mean_iat', 'fwd_mean_packet_size', 'bwd_mean_iat', 'bwd_mean_packet_size']
  D_i/D_rank_in_region  local KL contribution (higher = more specific to the target set)
  typicality           distance to region centre / region median (>>1 = atypical member)
  point_silhouette     how well the flow belongs to its region (-1..1)
  filter_marked/filter_score   flagged by the independent filter (method5/lof) used for B
  nearest_normal_cid/nearest_normal_dist   closest REFERENCE flow in the embedding
region_summary.csv     per region: service/protocol/endpoint mix, temporal-pattern mix,
                       oui_mix, frac_iot, frac_ech, direction_mix (new router logs),
                       sync_jaccard (0..1, flows active at the same moments), peak co-activity,
                       port/ip concentration, the 3 features most different from reference
region_contrast.csv    per region x feature: medians, Cohen's d, KS statistic/p vs reference
nearest_reference.csv  10 closest NORMAL flows to each region centre (side-by-side review)
regions/region_XX_R<id>.pdf   p1 overview + time-strip + overlay, p2 contrast + nearest normal,
                       p3.. ALL member flows (8 per page, every channel, absolute time)
review_sample.csv/.pdf blinded flagged+normal flows for manual labelling; answers in
review_key.csv         (do not show the key to the reviewer)
embeddings_proj.npz    XA/XB projected embeddings, cids, D_i, region labels
run_config.json        all arguments and counts
