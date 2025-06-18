# opensoundscapes

This library puts together a bunch of different models in an easy to use interface. This might end up being a fairly significant backbone for initial analysis.

Let's put together a list of models in a table, columns include file, start_time, and end_time.


| Model Name             | Clip Length (s) | Step Size (s) | Rows | Predict Columns | Embed Columns |
| :--------------------- | :-------------- | :------------ | :--- | :-------------- | :------------ |
| BirdNET                | 3               | 3             | 20   | 6522            | 1024          |
| YAMNet                 | 0.96            | 0.48          | 124  | 521             | 1024          |
| Perch                  | 5               | 5             | 12   | 10932           | 1280          |
| HawkEars               | 3               | 3             | 20   | 333             | 2048          |
| BirdSetConvNeXT        | 5               | 5             | 12   | 9736            | 1024          |
| BirdSetEfficientNetB1  | 5               | 5             | 12   | 9736            | 1280          |
| RanaSierraeCNN         | 2               | 2             | 30   | 2               | 512           |

To list all the available models:

```bash
python -m birdclef.infer.workflow list-models

{'BirdNET': bioacoustics_model_zoo.birdnet.BirdNET,
 'YAMNet': bioacoustics_model_zoo.yamnet.YAMNet,
 'Perch': bioacoustics_model_zoo.perch.Perch,
 'HawkEars': bioacoustics_model_zoo.hawkears.hawkears.HawkEars,
 'BirdSetConvNeXT': bioacoustics_model_zoo.bmz_birdset.bmz_birdset_convnext.BirdSetConvNeXT,
 'BirdSetEfficientNetB1': bioacoustics_model_zoo.bmz_birdset.bmz_birdset_efficientnetB1.BirdSetEfficientNetB1,
 'RanaSierraeCNN': bioacoustics_model_zoo.rana_sierrae_cnn.RanaSierraeCNN}
```

## timing

| Model | Inference Repr. | Total Time (s) | Files | Embeddings | Time/File (s) | Time/Embedding (s) |
| :---- | :-------------- | :------------- | :---- | :--------- | :------------ | :----------------- |
| Perch | TensorFlow      | 82.50          | 5     | 60         | 16.50         | 1.375              |
| Perch | TFLite (Flex)   | 8.43           | 5     | 60         | 1.69          | 0.141              |
