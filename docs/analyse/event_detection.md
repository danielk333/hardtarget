# Event detection
---
To find events in the measurements files it can be analysed with the event detection tool. This will run a
cross correlation over the data to find events.

The different supported libraries for event detection are:

::: hardtarget.types.constants.EventDetectionMethod

## Analysis
---
To simplify or rather clarify the usage of [analyse](api_analyse.md) for event detection a [wrapper](../reference/hardtarget/analyse.md#hardtarget.analyse.event_detection) is available that only supports **Event Detection**. More information about the running process can be found under [method/event detection](event_detection.md)

::: hardtarget.analyse.event_detection

## Process
---
::: hardtarget.matched_filter.xcorr.XCorrProcess

## Type specifics
---
::: hardtarget.matched_filter.xcorr.types