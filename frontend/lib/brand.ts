/** Product branding — single source of truth for the web app */
export const APP_NAME = "CurveScope"
export const APP_TAGLINE = "TESS light-curve anomaly lab"

/** Browser tab + SEO (keep under ~155 chars for descriptions where possible) */
export const APP_TITLE = `${APP_NAME} · ${APP_TAGLINE}`
export const APP_DESCRIPTION =
  "CurveScope finds unusual signals in NASA TESS photometry: GPU-trained autoencoder scores, MAST sector scans, BLS periods, wavelet vetting, centroid checks, and technosignature-style metrics—in one open-source workspace."

/** Short line for dashboards and empty states */
export const APP_ONE_LINER =
  "Scan TESS light curves from MAST, score anomalies with a trainable autoencoder, and vet candidates with periods, wavelets, and centroids."
