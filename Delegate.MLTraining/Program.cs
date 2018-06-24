using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;

namespace Delegate.MLTraining
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "train.csv");
        private static string TestDataPath => Path.Combine(AppPath, "test.csv");
        private static string ModelPath => Path.Combine(AppPath, "HousePriceModel.zip");

        static async Task Main(string[] args)
        {
            // STEP 1: Create a model
            var model = await TrainAsync();

            // STEP2: Test accuracy
            Evaluate(model);

            var prediction = model.Predict(new HousePrice()
            {
                LotArea = 8450,
                YearRemodAdd = 2003,
                YrSold = 2008,
                GrLivArea = 1710
            });
            Console.WriteLine($"Predicted SalePrice: {prediction.SalePrice:0.####}, actual SalePrice: 208500");

            using (var file = File.OpenText(TestDataPath))
            {

                var csv = new CsvReader(file);
                csv.Configuration.MemberTypes = CsvHelper.Configuration.MemberTypes.Fields;
                csv.Configuration.MissingFieldFound = null;
                csv.Configuration.HeaderValidated = null;
                var records = csv.GetRecords<HousePrice>();
                foreach(var r in records)
                {
                    var p = model.Predict(r);
                    Console.WriteLine($"{r.Id},{p.SalePrice.ToString(new CultureInfo(1033))}");
                }
            }



            Console.ReadLine();
        }

        private static async Task<PredictionModel<HousePrice, HousePricePrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.
            var pipeline = new LearningPipeline
            {
                // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
                // all the column names and their types.
                new TextLoader(TrainDataPath).CreateFrom<HousePrice>(useHeader: true, separator:','),
                
                // Transforms
                // When ML model starts training, it looks for two columns: Label and Features.
                // Label:   values that should be predicted. If you have a field named Label in your data type,
                //              no extra actions required.
                //          If you don't have it, like in this example, copy the column you want to predict with
                //              ColumnCopier transform:
                //new ColumnCopier(("SalePrice", "Label")),
                               

                new Microsoft.ML.Transforms.MinMaxNormalizer(
                    "LotArea"),
                // CategoricalOneHotVectorizer transforms categorical (string) values into 0/1 vectors
                
                //new CategoricalOneHotVectorizer("Id",
                //    "MSSubClass"),
                // Features: all data used for prediction. At the end of all transforms you need to concatenate
                //              all columns except the one you want to predict into Features column with
                //              ColumnConcatenator transform:
                new ColumnConcatenator("Features",
                nameof(HousePrice.YearRemodAdd),
                nameof(HousePrice.YrSold),
                nameof(HousePrice.GrLivArea),
                    "LotArea"),
                //FastTreeRegressor is an algorithm that will be used to train the model.
                new FastTreeRegressor()
                //new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "SalePrice" }
            };

            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<HousePrice, HousePricePrediction>();

            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<HousePrice, HousePricePrediction> model)
        {
            // To evaluate how good the model predicts values, it is run against new set
            // of data (test data) that was not involved in training.
            var testData = new TextLoader(TestDataPath).CreateFrom<HousePrice>(useHeader: true, separator: ',');

            // RegressionEvaluator calculates the differences (in various metrics) between predicted and actual
            // values in the test dataset.
            var evaluator = new RegressionEvaluator();

            Console.WriteLine("=============== Evaluating model ===============");

            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}, ideally should be around 2.8, can be improved with larger dataset");
            Console.WriteLine($"RSquared = {metrics.RSquared}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }
    }
}
