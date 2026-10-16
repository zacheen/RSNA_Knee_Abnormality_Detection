# Cache v5 settings. v4 geometry with the two non-FS slots dropped and their slice budget moved to the three FS slots,
# so a study still holds about 100 slices. A third-party ablation lost no AUC without the non-FS series.
V5 = {"RSNA_SLOTS": "3", "RSNA_N_SLICES": "32", "RSNA_IMG_SIZE": "320", "RSNA_CROP_MM": "128", "RSNA_BAND": "0.02,0.98"}
