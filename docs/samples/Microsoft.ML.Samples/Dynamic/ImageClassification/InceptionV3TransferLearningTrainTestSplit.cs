
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;
using System.Linq;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public class InceptionV3TransferLearningTrainTestSplit
    {
        public static void Example()
        {
            string assetsPath =// @"C:\Users\mzs\Downloads";
                @"E:\machinelearning-samples\samples\csharp\getting-started\DeepLearning_TensorFlow_TransferLearning\ImageClassification.Train\assets";

            //string imagesDownloadFolder = Path.Combine(assetsPath, "inputs", "images");
            string imagesFolder = //Path.Combine(assetsPath, "flower_photos");
            Path.Combine(assetsPath, "inputs", "images"); // "flower_photos"
            string imagesForPredictions = Path.Combine(assetsPath, "inputs", "images-for-predictions", "FlowersForPredictions");

            try
            {

                MLContext mlContext = new MLContext(seed: 1);

                //Load all the original images info
                IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: imagesFolder, useFolderNameasLabel: true);
                IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
                IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(fullImagesDataset);
                //shuffledFullImagesDataset = mlContext.Transforms.Conversion.MapValueToKey("Label")
                    //.Fit(shuffledFullImagesDataset)
                    //.Transform(shuffledFullImagesDataset);

                // Find the original label names.
                //VBuffer<ReadOnlyMemory<char>> keys = default;
                //shuffledFullImagesDataset.Schema["Label"].GetKeyValues(ref keys);
                //var originalLabels = keys.DenseValues().ToArray();

                // Split the data 80:20 into train and test sets, train and evaluate.
                TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.1, seed: 1);
                IDataView trainDataset = trainTestData.TrainSet;
                IDataView testDataset = trainTestData.TestSet;

                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label").Append(mlContext.Model.ImageClassification("ImagePath", "Label",
                            arch: ImageClassificationEstimator.Architecture.InceptionV3,
                            epoch: 100, //An epoch is one learning cycle where the learner sees the whole training data set.
                            batchSize: 100, // batchSize sets then number of images to feed the model at a time
                            learningRate: 0.01f,
                            metricsCallback: (metrics) => Console.WriteLine(metrics),
                            validationSet: null));//,
                            //reuseTrainSetBottleneckCachedValues: false,
                            //reuseValidationSetBottleneckCachedValues: false));


                Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

                // Measuring training time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var trainedModel = pipeline.Fit(trainDataset);

                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Training with transfer learning took: " + (elapsedMs / 1000).ToString() + " seconds");

                mlContext.Model.Save(trainedModel, trainDataset.Schema, "model.zip");

                ITransformer loadedModel;
                DataViewSchema schema;
                using (var file = File.OpenRead("model.zip"))
                    loadedModel = mlContext.Model.Load(file, out schema);

                EvaluateModel(mlContext, testDataset, loadedModel);

                VBuffer<ReadOnlyMemory<char>> keys = default;
                loadedModel.GetOutputSchema(schema)["Label"].GetKeyValues(ref keys);

                watch = System.Diagnostics.Stopwatch.StartNew();
                TrySinglePrediction(imagesForPredictions, mlContext, loadedModel, keys.DenseValues().ToArray());
                watch.Stop();
                elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Prediction engine took: " + (elapsedMs / 1000).ToString() + " seconds");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void TrySinglePrediction(string imagesForPredictions, MLContext mlContext, ITransformer trainedModel, ReadOnlyMemory<char>[] originalLabels)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ImageData, ImagePrediction>(trainedModel);

            IEnumerable<ImageData> testImages = LoadImagesFromDirectory(imagesForPredictions, false);
            ImageData imageToPredict = new ImageData
            {
                ImagePath = testImages.First().ImagePath
            };

            var prediction = predictionEngine.Predict(imageToPredict);
            var index = prediction.PredictedLabel;

            Console.WriteLine($"ImageFile : [{Path.GetFileName(imageToPredict.ImagePath)}], " +
                              $"Scores : [{string.Join(",", prediction.Score)}], " +
                              $"Predicted Label : {originalLabels[index]}");
        }


        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making bulk predictions and evaluating model's quality...");

            // Measuring time
            var watch2 = System.Diagnostics.Stopwatch.StartNew();

            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                              $"macro-accuracy = {metrics.MacroAccuracy}");

            watch2.Stop();
            long elapsed2Ms = watch2.ElapsedMilliseconds;

            Console.WriteLine("Predicting and Evaluation took: " + (elapsed2Ms / 1000).ToString() + " seconds");
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameasLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };

            }
        }

        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        public class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public UInt32 PredictedLabel;
        }
    }
}

