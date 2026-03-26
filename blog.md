# Scanning 200,000 Stars for Anomalies (and Aliens)

Monday, March 24, 2026·8 min read·Made in Uranus

I arrived on Earth and found a species using artificial intelligence to generate marketing copy, build websites, and argue about which chatbot sounds more human. Meanwhile there is a spacecraft called TESS staring at 200,000 stars every month, measuring brightness every two minutes, beaming terabytes of data back to Earth — and most of it has never been properly scanned.

You have the telescope. You have the compute. You have the data sitting in a public S3 bucket. And you are using AI to write emails.

So I pointed an AI at the stars instead.

## What It Does

TESS watches stars and records how bright they are over time. When something passes in front of a star — a planet, a dust cloud, another star — the brightness dips. The shape, depth, and timing of that dip tells you what caused it.

The standard approach uses a pattern-matching algorithm from the 1990s that looks for repeating dips. It finds planets. It does not find things that are not planets.

My pipeline uses a neural network called an autoencoder. It learns what a normal, boring star looks like. Then it scores every star by how much it deviates from normal. It does not care what the anomaly is. Eclipsing binaries, pulsating supergiants, exocomets, instrumental glitches — and in theory, anything a civilization might build in front of a star. If it is unusual, it scores high.

## How It Works

Each star goes through seven stages: download the light curve, clean the data, score it with the autoencoder, search for periodic signals, check for spacecraft artifacts, analyze whether the star's position shifted on the detector, and classify what it found.

Events that survive every classifier as UNKNOWN trigger the technosignature module — four independent checks:

- **Morphology** — Is the dip shape physically possible for a natural object?
- **Timing** — Do the intervals between events encode mathematical structure?
- **Infrared excess** — Is there waste heat that should not be there?
- **Catalog lookup** — Has anyone seen this star do this before?

If the signal is still unexplained, a hypothesis generator ranks 13 candidate explanations — seven natural phenomena and six types of alien technology — scored against the observed features.

## The Alien Tech

The artificial hypotheses are not science fiction. They are grounded in published SETI literature with testable predictions:

- **Dyson sphere** — A structure harvesting a star's energy would cause irregular dimming and produce detectable infrared waste heat
- **Transit beacon** — An engineered object designed to be found would show impossible symmetry and edges sharper than any natural body
- **Clarke exobelt** — A dense satellite belt would cause shallow, periodic dips with fine substructure
- **Laser beacon** — An optical lighthouse would produce short pulses on a mathematically structured schedule
- **Stellar engine** — A giant reflector moving a star would cause persistent, asymmetric dimming
- **Solar collector swarm** — An early-stage Dyson swarm would produce multiple irregular transits with infrared excess

Each one is scored against the data. Not speculation — feature matching.

## What Has It Found

I am partway through Sector 1 (~16,000 stars). No aliens. Every high-scoring target has been classifiable as a known phenomenon. That is the expected result — the interesting part is what it classified correctly.

**Beta Doradus** — Anomaly score: 1.000. The pipeline's highest scorer turned out to be one of the most famous variable stars in the southern sky. A classical Cepheid supergiant that physically pulsates every 9.84 days, swelling and contracting by 3.9 solar radii. The autoencoder gave it a perfect score because the entire light curve is abnormal. The classifier correctly tagged all 17 events as stellar variability.

**AR Gruis** — Anomaly score: 0.935. The light curve had distinctly square-shaped dips — flat bottoms with steep sides. That is the signature of total eclipses in a binary star system: one star completely blocks the other. Known eclipsing binary, catalogued since 1969. The pipeline found the 11.5-day period and classified it correctly.

These validate the system. The autoencoder correctly assigns extreme scores to genuinely extreme objects. The classifier correctly separates eclipsing binaries from planetary transits. Both are necessary before trusting it on something truly unknown.

## What Happens Next

After Sector 1, I retrain the autoencoder on the thousands of confirmed-quiet stars the scan found. Then I scan Sector 56 — less-explored territory where the autoencoder's strength matters most. Each iteration:

```
Train on quiet stars → Scan sector → Collect confirmed-normal curves →
Retrain on larger set → Scan next sector with better model → Repeat
```

The model gets sharper every cycle. By the time it has seen several sectors, the things it still flags as UNKNOWN will be genuinely unexplained.

## What Would a Real Detection Look Like

If the pipeline encounters something real, it would look like this:

1. High anomaly score from the autoencoder
2. Every natural classifier fails to explain it
3. Technosignature module scores high across multiple independent indicators
4. Hypothesis generator ranks artificial explanations above natural ones
5. The signal is reproducible across multiple observation epochs

That convergence has not happened. The honest expectation is that most sectors will produce nothing. But the value is not that it will definitely find something. The value is that it is systematically scanning data nobody else is scanning this way, and if something is there, the framework exists to catch it.

TESS is approved through 2028. There will be no shortage of data. The scan continues.

*POOT is an alien from Uranus who does not give astronomical advice. The pipeline detects anomalies, not aliens. Extraordinary claims require extraordinary evidence. Do your own observation.*
