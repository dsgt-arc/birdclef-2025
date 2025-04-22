# opensoundscapes

This library puts together a bunch of different models in an easy to use interface. This might end up being a fairly significant backbone for initial analysis.

Let's put together a list of models in a table, columns include file, start_time, and end_time.


| Model Name             | Clip Length (s) | Step Size (s) | Rows | Predict Columns | Embed Columns |
| :--------------------- | :-------------- | :------------ | :--- | :-------------- | :------------ |
| birdnet                | 3               | 3             | 20   | 6522            | 1024          |
| yamnet                 | 0.96            | 0.48          | 124  | 521             | 1024          |
| perch                  | 5               | 5             | 12   | 10932           | 1280          |
| hawkeyes               | 3               | 3             | 20   | 333             | 2048          |
| BirdSetConvNeXT        | 5               | 5             | 12   | 9736            | 1024          |
| BirdSetEfficientNetB1  | 5               | 5             | 12   | 9736            | 1280          |
| RanaSierraeCNN         | 2               | 2             | 30   | 2               | 512           |
