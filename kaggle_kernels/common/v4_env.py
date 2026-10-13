# Cache v4 settings, shared by every v4 kernel so the values stay identical. 5 slots of 20 slices spread over the
# central 96% of each stack, a 128 mm physical centre crop resized to 320 px (0.4 mm per pixel). WindowNet crops 288.
V4 = {"RSNA_SLOTS": "5", "RSNA_N_SLICES": "20", "RSNA_IMG_SIZE": "320", "RSNA_CROP_MM": "128", "RSNA_BAND": "0.02,0.98"}
