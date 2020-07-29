package demo.common

import java.util.Properties
import scala.collection.JavaConversions._
import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SaveMode
import java.util.Date
import java.util.Calendar
import java.util.Iterator;
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.isInstanceOf
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.storage.StorageLevel
import java.text.SimpleDateFormat
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive._
import org.apache.spark.sql.SparkSession
import java.sql.{ Timestamp, Date }
import demo.utils.DataframeReadWriteUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types._
import demo.constants.Constants
import demo.helper.ProcessDataHelper
import org.apache.log4j.{ Level, Logger }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{ RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.PCAModel

object WineQualityModelTrain {

  def main(args: Array[String]) {

    if (args.isEmpty) {
      println(Constants.NO_ARGUMENT_MSG)
      System.exit(0);
    }
    val params = args.map(_.split('=')).map {
      case Array(param, value) => (param, value)
    }.toMap

    var trainingDataPath: String = ""
    var validationDataPath: String = ""
    var outputPath: String = ""
    var master: String = ""
    var runType: String = ""
    var testingDataPath: String = ""

    if (params.contains("--run-type")) {
      runType = params.get("--run-type").get.asInstanceOf[String]
    } else {
      println("--run-type Is Missing In The Argument")
      System.exit(0);
    }

    if (params.contains("--master")) {
      master = params.get("--master").get.asInstanceOf[String]
    } else {
      println("--master Is Missing In The Argument")
      System.exit(0);
    }

    if (runType == "training") {

      if (params.contains("--training-file-path")) {
        trainingDataPath = params.get("--training-file-path").get.asInstanceOf[String]
      } else {
        println("--training-file-path Is Missing In The Argument")
        System.exit(0);
      }

      if (params.contains("--validation-file-path")) {
        validationDataPath = params.get("--validation-file-path").get.asInstanceOf[String]
      } else {
        println("--validation-file-path Is Missing In The Argument")
        System.exit(0);
      }

    } else if (runType == "prediction") {
      if (params.contains("--testing-file-path")) {
        testingDataPath = params.get("--testing-file-path").get.asInstanceOf[String]
      } else {
        println("--testing-file-path Is Missing In The Argument")
        System.exit(0);
      }
    }

    if (params.contains("--model-store-path")) {
      outputPath = params.get("--model-store-path").get.asInstanceOf[String]
    } else {
      println("--output-file-path Is Missing In The Argument")
      System.exit(0);
    }

    val sparkSession = SparkSession.builder
      .appName(s"WineQuality$runType")
      .master(s"$master")
      .getOrCreate
    try {
      import sparkSession.implicits._
      val sc = sparkSession.sparkContext
      val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
/***********************************************************************************
       * below should be uncommented when you run  from local.Give S3 credentials as well
       ***********************************************************************************/
      //      sc.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      //      sc.hadoopConfiguration.set("fs.s3a.access.key", "")
      //      sc.hadoopConfiguration.set("fs.s3a.secret.key", "")
      //      sc.hadoopConfiguration.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
      //      sc.hadoopConfiguration.set("fs.s3a.session.token", "")
      //      sc.hadoopConfiguration.set("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")
      //      sc.hadoopConfiguration.set("com.amazonaws.services.s3.enableV4", "true")

      //
      val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")
      if (runType == "training") {
        val trainingDf = ProcessDataHelper.readWineData(sparkSession, trainingDataPath)
        val validationDf = ProcessDataHelper.readWineData(sparkSession, validationDataPath)

/*****************************************************************************************
       *  					Logistic Classification Model Testing
       *  
       *****************************************************************************************/

        val createLogisticRegModelPipeline = new Pipeline().setStages(Array(ProcessDataHelper.encodeStringIndex(), ProcessDataHelper.assembler(), ProcessDataHelper.standardizer(), ProcessDataHelper.LogisticModelCreation))
        val logisticModelFit = ProcessDataHelper.trainModel(createLogisticRegModelPipeline, trainingDf)
        val logisticModelTrainpredictions = logisticModelFit.transform(trainingDf)
        val logisticModelTrainaccuracy = evaluator.evaluate(logisticModelTrainpredictions)
        val logisticModelpredictions = logisticModelFit.transform(validationDf)
        val logisticModelaccuracy = evaluator.evaluate(logisticModelpredictions)
        println("Logistic Model Accuracy For Training Dataset -->", logisticModelTrainaccuracy)
        println("Logistic Model Accuracy For Validation Dataset -->", logisticModelaccuracy)

/*****************************************************************************************
       *  					Decision Tree Model Testing
       *  
       *****************************************************************************************/
        val decisionTreeModelPipeline = new Pipeline().setStages(Array(ProcessDataHelper.encodeStringIndex(), ProcessDataHelper.assembler(), ProcessDataHelper.standardizer(), ProcessDataHelper.decisionTreeClassification()))
        val decisionTreeModelFit = ProcessDataHelper.trainModel(decisionTreeModelPipeline, trainingDf)
        val decisionTreepredictions = decisionTreeModelFit.transform(validationDf)
        val decisionTreeAccuracy = evaluator.evaluate(decisionTreepredictions)
        println("Decision Tree Model Accuracy -->", decisionTreeAccuracy)

/*****************************************************************************************
       *  					Random forest Model Testing
       *  
       *****************************************************************************************/
        val createRandomForestModelPipeline = new Pipeline().setStages(Array(ProcessDataHelper.encodeStringIndex(), ProcessDataHelper.assembler(), ProcessDataHelper.standardizer(), ProcessDataHelper.randomForestClassification()))
        val randomForestModelFit = ProcessDataHelper.trainModel(createRandomForestModelPipeline, trainingDf)
        val rFTrainpredictions = randomForestModelFit.transform(trainingDf)
        val rFpredictions = randomForestModelFit.transform(validationDf)
        val rFtrainaccuracy = evaluator.evaluate(rFTrainpredictions.select("prediction", "label"))
        val rFaccuracy = evaluator.evaluate(rFpredictions.select("prediction", "label"))
        println("Random Forest Model Accuracy For Training Dataset Without Model Tuning -->", rFtrainaccuracy)
        println("Random Forest Model Accuracy For Validation Dataset Without Model Tuning -->", rFaccuracy)
        randomForestModelFit.write.overwrite.save(s"$outputPath")

        //      println("random forest model -->",ProcessDataHelper.trainModel(createRandomForestModelPipeline, trainingDf).stages(3).asInstanceOf[RandomForestClassificationModel].toDebugString)
        //      println("feature importance -->",randomForestModelFit.stages(3).asInstanceOf[RandomForestClassificationModel].featureImportances)
        ////
        println("************************************************************************************************")
        println("*                              Model Tuning Starts                                             *")
        println("************************************************************************************************")

        //***************************************************
        //
        // Model Tuning Starts here
        //
        //***************************************************
        val crossVaidatorTrain = ProcessDataHelper.crossValidatorTuning(createRandomForestModelPipeline, trainingDf)
        val rFTrainpredictionsTuning = crossVaidatorTrain.transform(trainingDf)
        val rFpredictionsTuning = crossVaidatorTrain.transform(validationDf)
        val rFTrainaccuracyTuning = evaluator.evaluate(rFTrainpredictionsTuning.select("prediction", "label"))
        val rFaccuracyTuning = evaluator.evaluate(rFpredictionsTuning.select("prediction", "label"))
        println("Random Forest Model Accuracy  For Training Dataset With Model Tuning -->", rFTrainaccuracyTuning)
        val metrics = new BinaryClassificationMetrics(rFpredictionsTuning.select("prediction", "label").as[(Double, Double)].rdd)
        println("*************************Validation Dataset Metrics********************************")
        println("Random Forest Model Accuracy  For Validation Dataset With Model Tuning -->", rFaccuracyTuning)
        println("Area under ROC -->", metrics.areaUnderROC)
        println("Area under PR -->", metrics.areaUnderPR())

        ProcessDataHelper.calculateMetrics(sparkSession, rFpredictionsTuning)

      } else if (runType == "prediction") {
        val loadedModel = PipelineModel.load(s"$outputPath")
        val testDf = ProcessDataHelper.readWineData(sparkSession, testingDataPath)
        val predictionsTuning = loadedModel.transform(testDf)
        val accuracy = evaluator.evaluate(predictionsTuning.select("prediction", "label"))
        println("Test Dataset Accuracy -->", accuracy)
        ProcessDataHelper.calculateMetrics(sparkSession, predictionsTuning)
      }

      sparkSession.stop
    } catch {
      case e: Exception =>
        val builder = StringBuilder.newBuilder
        builder.append(e.getMessage)
        (e.getStackTrace.foreach { x => builder.append(x + "\n") })
        val err_message = builder.toString()
        println(err_message)
        sparkSession.stop()

    }
  }
}