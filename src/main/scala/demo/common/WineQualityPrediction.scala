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
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{RandomForestClassificationModel,RandomForestClassifier,LogisticRegression}
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.PCAModel

object WineQualityPrediction {

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
    if (params.contains("--training-file-path")) {
      trainingDataPath = params.get("--training-file-path").get.asInstanceOf[String]
    } else {
      println(Constants.INVALID_KEY)
      System.exit(0);
    }

    if (params.contains("--validation-file-path")) {
      validationDataPath = params.get("--validation-file-path").get.asInstanceOf[String]
    } else {
      println(Constants.INVALID_KEY)
      System.exit(0);
    }

    if (params.contains("--output-file-path")) {
      outputPath = params.get("--output-file-path").get.asInstanceOf[String]
    } else {
      println(Constants.INVALID_KEY)
      System.exit(0);
    }

    if (params.contains("--master")) {
      master = params.get("--master").get.asInstanceOf[String]
    } else {
      println(Constants.INVALID_KEY)
      System.exit(0);
    }

    val sparkSession = SparkSession.builder
      .appName("WineQualityPrediction")
      .master(s"$master")
      .getOrCreate
    try {
      import sparkSession.implicits._
      val sc = sparkSession.sparkContext
      val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
      sc.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      sc.hadoopConfiguration.set("fs.s3a.access.key", "")
      sc.hadoopConfiguration.set("fs.s3a.secret.key", "")
      sc.hadoopConfiguration.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
      sc.hadoopConfiguration.set("fs.s3a.session.token", "")
      sc.hadoopConfiguration.set("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")
      sc.hadoopConfiguration.set("com.amazonaws.services.s3.enableV4", "true")
      
      val trainingDf = ProcessDataHelper.readWineData(sparkSession, trainingDataPath)
      val validationDf = ProcessDataHelper.readWineData(sparkSession, validationDataPath)
      val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
      
      /*****************************************************************************************
       *  					Logistic Classification Model Testing
       *  
       *****************************************************************************************/
      val createLogisticRegModelPipeline = new Pipeline().setStages(Array(ProcessDataHelper.encodeStringIndex(),ProcessDataHelper.assembler(),ProcessDataHelper.standardizer(),ProcessDataHelper.LogisticModelCreation))
      val logisticModelFit = ProcessDataHelper.trainModel(createLogisticRegModelPipeline, trainingDf)
      logisticModelFit.transform(validationDf).show(20)
      val logisticModelpredictions=logisticModelFit.transform(validationDf)
      val logisticModelaccuracy = evaluator.evaluate(logisticModelpredictions)
      println("Logistic Model Accuracy-->",logisticModelaccuracy)

      
       /*****************************************************************************************
       *  					Decision Tree Model Testing
       *  
       *****************************************************************************************/
      val decisionTreeModelPipeline = new Pipeline().setStages(Array(ProcessDataHelper.encodeStringIndex(),ProcessDataHelper.assembler(),ProcessDataHelper.standardizer(),ProcessDataHelper.decisionTreeClassification()))
      val decisionTreeModelFit = ProcessDataHelper.trainModel(decisionTreeModelPipeline, trainingDf)
      val decisionTreepredictions=decisionTreeModelFit.transform(validationDf)
      val decisionTreeAccuracy = evaluator.evaluate(decisionTreepredictions)
      println("Decision Tree Model Accuracy -->",decisionTreeAccuracy)
      

      /*****************************************************************************************
       *  					Random forest Model Testing
       *  
       *****************************************************************************************/
      val createRandomForestModelPipeline = new Pipeline().setStages(Array(ProcessDataHelper.encodeStringIndex(),ProcessDataHelper.assembler(),ProcessDataHelper.standardizer(),ProcessDataHelper.randomForestClassification()))
      val randomForestModelFit = ProcessDataHelper.trainModel(createRandomForestModelPipeline, trainingDf)
      val rFpredictions=randomForestModelFit.transform(validationDf)
      val rFaccuracy = evaluator.evaluate(rFpredictions)
      println("Random Forest Model Accuracy Without Model Tuning -->",rFaccuracy)
      println("random forest model -->",ProcessDataHelper.trainModel(createRandomForestModelPipeline, trainingDf).stages(3).asInstanceOf[RandomForestClassificationModel].toDebugString)
      println("feature importance -->",randomForestModelFit.stages(3).asInstanceOf[RandomForestClassificationModel].featureImportances)
    
      //***************************************************  
      // 
      // Model Tuning Starts here
      // 
      //***************************************************
      val crossVaidatorTrain = ProcessDataHelper.crossValidatorTuning(createRandomForestModelPipeline, trainingDf)
      val rFpredictionsTuning = crossVaidatorTrain.transform(validationDf)
      val rFaccuracyTuning = evaluator.evaluate(rFpredictionsTuning)
      println("Random Forest Model Accuracy Without Model Tuning -->",rFaccuracyTuning)

      

//      val predictionAndLabels = randomForestModelFit.transform(validationDf).select("prediction", "label").as[(Double, Double)].rdd
//      val metrics = new MulticlassMetrics(predictionAndLabels)
//      val labels = metrics.labels
//      println("Confusion matrix:")
//      println(metrics.confusionMatrix)
//      println("accuracy:")
//      println(metrics.accuracy)
//      // Precision by label
//      labels.foreach { l =>
//        println(s"Precision($l) = " + metrics.precision(l))
//      }
//
//      // Recall by label
//      labels.foreach { l =>
//        println(s"Recall($l) = " + metrics.recall(l))
//      }
//
//      // False positive rate by label
//      labels.foreach { l =>
//        println(s"FPR($l) = " + metrics.falsePositiveRate(l))
//      }
//
//      // F-measure by label
//      labels.foreach { l =>
//        println(s"F1-Score($l) = " + metrics.fMeasure(l))
//      }
//
////      println(s"Random Forest --> RMSE on traning data = $trainingPrediction")
////      println(s"Random Forest --> RMSE on test data = $validationPrediction")
//      // Weighted stats
//      println(s"Weighted precision: ${metrics.weightedPrecision}")
//      println(s"Weighted recall: ${metrics.weightedRecall}")
//      println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
//      println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
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