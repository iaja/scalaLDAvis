package com.iaja.ldavis

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDAModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.reflect.runtime.universe._
//import scala.language.implicitConversions

/**
  * Created by mageswarand on 5/6/17.
  */

abstract class LDAvis {
  /**
    * Transforms the topic model distributions and related corpus data into
    * the data structures needed for the visualization.
    * Creates the directory if not existing and copy the necessary files for
    * visualization.
    * Open the 'index.html' in Firefox for the visualization to work seamlessly!
    * @param directory Where you wanted to store the visualizations
    * @param lambdaStep Determines the interstep distance in the grid of lambda values over
    *                    which to iterate when computing relevance.
    *                    Default is 0.01. Recommended to be between 0.01 and 0.1.
    * @param plotOpts Map[String, String], with keys 'xlab' and `ylab`
    *                  Dictionary of plotting options, right now only used for the axis labels.
    * @param R Number of topics to be shown on the UI. Recomended is 20 to 50
    */
  def prepareLDAVisData(directory: String,
                        lambdaStep: Double = 0.01,
                        plotOpts: Map[String, String] = Map("xlab" -> "PC1", "ylab" -> "PC2"),
                        R:Int =30): Unit
}

case class LDAVisData()

abstract class LDAvisBuilder {

  var spark: Option[SparkSession] = None
  /**
    * Pass when you have existing SparkSession or simply leave it to run locally
    * @param spark SparkSession
    * @return LDAvisBuilder
    */
  def withSparkSession(spark: SparkSession): LDAvisBuilder

  //fitted or transformed or trained DataFrame that has
  // 'topicDistribution' or equivalent column created with LDA transformer
  var transformedDfPath: Option[String] = None
  var transformedDf: Option[DataFrame] = None

  /**
    * Offline - Load the stored fitted/transformed dataframe (i.e., pipeLineModel.transform(df))
    * @param path Directory to the stored dataframe
    * @return LDAvisBuilder
    */
  def withTransformedDfPath(path: String): LDAvisBuilder
  /**
    * Online - Pass the fitted/transformed dataframe (i.e., pipeLineModel.transform(df))
    * @param df Transformed dataframe with ML pipeline model
    * @return LDAvisBuilder
    */
  def withTransformedDf(df: DataFrame): LDAvisBuilder

  //LDA Model - Runtime or from stored model
  var ldaModelPath: Option[String] = None
  var ldaModel: Option[LDAModel] = None
  /**
    * Online - Pass the trained LDAModel
    * @param model LDAModel
    * @return LDAvisBuilder
    */
  def withLDAModel(model: LDAModel): LDAvisBuilder
  /**
    * Offline - Load the trained LDA model
    * @param path Directory to the stored LDAModel
    * @return LDAvisBuilder
    */
  def withLDAModelPath(path: String): LDAvisBuilder

  //CountVectorizer Model - Runtime or from stored model
  var cvModelPath: Option[String] = None
  var cvModel: Option[CountVectorizerModel] = None
  var cvOutCol: Option[String] = None //Retreived from CV model
  /**
    * Online - Pass the trained CountVectorizerModel
    * @param model CountVectorizerModel
    * @return LDAvisBuilder
    */
  def withCVModel(model: CountVectorizerModel): LDAvisBuilder
  /**
    * Offline - Load the trained CountVectorizerModel model
    * @param path Directory to the stored CountVectorizerModel
    * @return LDAvisBuilder
    */
  def withCVModelPath(path: String): LDAvisBuilder

  //Vocabulary - Runtime or from stored model
  var vocabDFPath: Option[String] = None
  var vocab: Array[String] = Array()
  /**
    * List of all the words in the corpus used to train the model.
    * @param words Array[String]
    * @return LDAvisBuilder
    */
  def withVocab(words: Array[String]): LDAvisBuilder
//  def withVocabDfPath(path: String): LDAvisBuilder


  //LDA pipeline runtime
//  var ldaPipeline: Option[Pipeline] = None
//  def withLDAPipeline(pipeline: Pipeline): LDAvisBuilder

  var debugFlag: Boolean = false
  /**
    * Enable or Disable debug
    * @param flag Booelan false -> Disable, true -> Enable
    * @return
    */
  def withEnableDebug(flag: Boolean): LDAvisBuilder

  def build: LDAvis
}

//--------------------------------------------------------------------------------------
//TODO find a better way to convert the columns of arrays to DataFrame
case class DefaultTermInfo(Saliency: Double, Term: String,
                           Freq: Double, Total:Double, Category: String)

case class ZippedDefaultTermInfo(Saliency: Array[Double], Term: Array[String],
                                 Freq: Array[Double], Total: Array[Double], Category: Array[String]) {

  def toDefaultTermInfoArray(R:Int) = {
    assert((Saliency.length == Term.length &&
      Saliency.length == Freq.length &&
      Saliency.length == Total.length &&
      Saliency.length == Category.length),
      "Length of all arrays should be same!"
    )

    val length = Saliency.length
    // Rounding Freq and Total to integer values to match LDAvis code:
    (0 until length).map(i => DefaultTermInfo(Saliency(i), Term(i),
      Math.floor(Freq(i)), Math.floor(Total(i)), Category(i)))
  }
}

//--------------------------------------------------------------------------------------

case class TopicTopTermRow(Term: String, Freq: Double, Total: Double, Category: String,
                           loglift: Double, logprop: Double)

case class ZippedTopicTopTermRows(Term: Array[String], Freq: Array[Double], Total: Array[Double],
                                  logprob: Array[Double], loglift: Array[Double], Category: Array[String]) {
  def toTopicTopTermRow() = {

    assert((Term.length == Freq.length &&
      Term.length == Total.length &&
      Term.length == logprob.length &&
      Term.length == loglift.length &&
      Term.length == Category.length), "Length of all arrays should be same!" +
      Term.length + "," +
      Total.length + "," +  logprob.length + "," +  loglift.length + "," + Category.length
    )

    val length = Term.length
    (0 until length).map(i => TopicTopTermRow(Term(i), Freq(i), Total(i), Category(i), loglift(i), logprob(i)))
  }
}

case class ZippedTopicInfo(Category:Array[String],	Freq:Array[Double],	Term:Array[String],
                           Total:Array[Double],	loglift:Array[Double],	logprob:Array[Double])

//--------------------------------------------------------------------------------------
case class TokenTable(TermId: Int, Topic: Int, Freq: Double, Term: String)



case class ZippedTokenTable(TermId: Array[Int], Topic: Array[Int], Freq: Array[Double], Term: Array[String])

case class TopicCoordinates(x: Double, y:Double, topics: Int, cluster: Int, Freq: Double)

case class ZippedTopicCoordinates(x: Array[Double], y:Array[Double], topics: Array[Int], cluster: Array[Int], Freq: Array[Double])
