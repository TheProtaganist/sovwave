# Sovwave Cross-Language MKV Runtimes

Run inference from a trained Sovwave `.mkv` model in **Python, C++, JavaScript, Java, or C#** — no Julia required.

All runtimes share the same protocol:
1. Decode the **FFV1 data stream** (`0:v:1`) from the MKV via `ffmpeg`
2. Map pixel values back to wave parameters: `amp = (R/255)×2`, `phase = (G/255)×2π`, `freq = (B/255)×4`
3. Run the **Sovwave wave forward pass**: fractal dimension + wave speed aware

---

## Prerequisites

- **ffmpeg** must be in your system `PATH` (all runtimes use it to extract the data stream)
- The `_meta.yaml` companion file (saved alongside the `.mkv`) is optional but recommended for correct layer dimensions

---

## Python (3.8+, stdlib only)

```bash
python sovwave_runtime.py my_model.mkv --input 0.1,0.2,0.3,0.4
```

```python
from sovwave_runtime import SovwaveModel

model   = SovwaveModel.load("my_model.mkv", "my_model_meta.yaml")
outputs = model.predict([[0.1, 0.2, 0.3, 0.4]])
print(outputs[0][:4])
```

> **Optional:** `pip install pyyaml` for richer YAML config parsing. Works without it using defaults.

---

## C++17 (header-only)

```cpp
#include "sovwave_runtime.hpp"

auto model  = sovwave::SovwaveModel::load("my_model.mkv");
auto output = model.predict({{0.1, 0.2, 0.3, 0.4}});
for (double v : output[0]) std::cout << v << " ";
```

Compile with any C++17 compiler:
```bash
g++ -std=c++17 -O2 -o my_app my_app.cpp
```

---

## JavaScript (Node.js 14+ / Browser)

```javascript
// Node.js
const { SovwaveModel } = require('./sovwave_runtime');
const model   = await SovwaveModel.load('my_model.mkv');
const outputs = model.predict([[0.1, 0.2, 0.3, 0.4]]);
console.log(outputs[0].slice(0, 4));

// Browser (pre-extracted raw RGB ArrayBuffer)
const model   = await SovwaveModel.fromArrayBuffer(mkvRawBuffer, metaYamlText);
const outputs = model.predict([[0.1, 0.2, 0.3, 0.4]]);
```

---

## Java (11+)

```bash
javac SovwaveRuntime.java
java SovwaveModel my_model.mkv my_model_meta.yaml 0.1,0.2,0.3,0.4
```

```java
SovwaveModel model   = SovwaveModel.load("my_model.mkv", "my_model_meta.yaml");
double[][]   outputs = model.predict(new double[][]{{ 0.1, 0.2, 0.3, 0.4 }});
System.out.println(Arrays.toString(outputs[0]));
```

---

## C# (.NET 6+)

```bash
dotnet run my_model.mkv my_model_meta.yaml 0.1,0.2,0.3,0.4
```

```csharp
var model   = await SovwaveModel.LoadAsync("my_model.mkv", "my_model_meta.yaml");
var outputs = model.Predict(new[] { new[] { 0.1, 0.2, 0.3, 0.4 } });
Console.WriteLine(string.Join(", ", outputs[0]));
```

---

## Pixel → Weight Protocol

Each MKV data stream frame encodes one WaveLayer:

| Channel | Encoded value | Decode formula |
|---------|--------------|----------------|
| Red (R) | Amplitude    | `amp = (R / 255) × 2.0` |
| Green (G) | Phase      | `phase = (G / 255) × 2π` |
| Blue (B)  | Frequency  | `freq = max(0.1, (B / 255) × 4.0)` |

> **fractal_dim** and **wave_speed** defaults (`1.5` and `1.0`) are used by all runtimes unless extended metadata is present in future versions.

---

## Wave Speed Notes

- **Default speed** (`1.0`): natural unit propagation — `angle = ω·f·(x/v) + φ - t`
- **Unlimited** (`-1.0`): instantaneous propagation — `angle = ω·f·x + φ - t` (no speed divisor)
- Wave speed is a per-node property in the Julia model; runtimes default all nodes to `1.0`
