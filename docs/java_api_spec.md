# Java Integration Spec: PvP Inference API

## Endpoint and Tick Timing

The inference server exposes:

- `POST http://127.0.0.1:8000/predict`

Your Java bot integration should send **one request every 50 ms** (one Minecraft tick) per bot.

## Request Schema (`GameState`)

Send JSON with the exact fields below.

```json
{
  "type": "object",
  "required": [
    "bot_id",
    "health",
    "foodLevel",
    "damageDealt",
    "velX",
    "velY",
    "velZ",
    "yaw",
    "pitch",
    "isOnGround",
    "targetDistance",
    "targetRelX",
    "targetRelY",
    "targetRelZ",
    "nearestProjectileDx",
    "nearestProjectileDy",
    "nearestProjectileDz",
    "targetYaw",
    "targetPitch",
    "targetVelX",
    "targetVelY",
    "targetVelZ",
    "targetHealth",
    "mainHandItem",
    "offHandItem",
    "hotbar0",
    "hotbar1",
    "hotbar2",
    "hotbar3",
    "hotbar4",
    "hotbar5",
    "hotbar6",
    "hotbar7",
    "hotbar8",
    "inventoryBag"
  ],
  "properties": {
    "bot_id": { "type": "string" },

    "health": { "type": "number" },
    "foodLevel": { "type": "number" },
    "damageDealt": { "type": "number" },
    "velX": { "type": "number" },
    "velY": { "type": "number" },
    "velZ": { "type": "number" },
    "yaw": { "type": "number" },
    "pitch": { "type": "number" },
    "isOnGround": { "type": "boolean" },
    "targetDistance": { "type": "number" },
    "targetRelX": { "type": "number" },
    "targetRelY": { "type": "number" },
    "targetRelZ": { "type": "number" },
    "nearestProjectileDx": { "type": "number" },
    "nearestProjectileDy": { "type": "number" },
    "nearestProjectileDz": { "type": "number" },
    "targetYaw": { "type": "number" },
    "targetPitch": { "type": "number" },
    "targetVelX": { "type": "number" },
    "targetVelY": { "type": "number" },
    "targetVelZ": { "type": "number" },
    "targetHealth": { "type": "number" },

    "mainHandItem": { "type": "string" },
    "offHandItem": { "type": "string" },
    "hotbar0": { "type": "string" },
    "hotbar1": { "type": "string" },
    "hotbar2": { "type": "string" },
    "hotbar3": { "type": "string" },
    "hotbar4": { "type": "string" },
    "hotbar5": { "type": "string" },
    "hotbar6": { "type": "string" },
    "hotbar7": { "type": "string" },
    "hotbar8": { "type": "string" },

    "inventoryBag": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 27,
      "maxItems": 27
    }
  },
  "additionalProperties": false
}
```

## Response Schema (`BotPrediction`)

The API returns JSON with the exact fields below.

```json
{
  "type": "object",
  "required": [
    "deltaYaw",
    "deltaPitch",
    "inputForward",
    "inputBackward",
    "inputLeft",
    "inputRight",
    "inputJump",
    "inputSneak",
    "inputSprint",
    "inputLmb",
    "inputRmb",
    "inputSlot"
  ],
  "properties": {
    "deltaYaw": { "type": "number" },
    "deltaPitch": { "type": "number" },
    "inputForward": { "type": "number" },
    "inputBackward": { "type": "number" },
    "inputLeft": { "type": "number" },
    "inputRight": { "type": "number" },
    "inputJump": { "type": "number" },
    "inputSneak": { "type": "number" },
    "inputSprint": { "type": "number" },
    "inputLmb": { "type": "number" },
    "inputRmb": { "type": "number" },
    "inputSlot": { "type": "integer", "minimum": 0, "maximum": 8 }
  },
  "additionalProperties": false
}
```

## Mandatory Java Concurrency Requirement

Use `java.net.http.HttpClient` (Java 11+) **asynchronously**.

- Do **not** block the Minecraft main server thread with a synchronous HTTP call.
- Use `sendAsync(...)` and `thenAccept(...)`.
- In `thenAccept(...)`, schedule application of the returned prediction onto the Citizens NPC on the **next available tick** via Bukkit scheduler (for example, `Bukkit.getScheduler().runTask(...)`).

This is required to avoid tick lag and server thread stalls.

