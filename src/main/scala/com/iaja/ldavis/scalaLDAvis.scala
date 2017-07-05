package com.iaja.ldavis

import breeze.linalg.{Axis, Transpose, sum, DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.numerics.log
import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.{LDAModel, LocalLDAModel}
import org.apache.spark.ml.linalg.{Vector => MLVector, _}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StructField, StructType}

import scala.collection.mutable
import com.iaja.ldavis._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.CountVectorizerModel

import scala.annotation.tailrec

/**
  * Created by mageswarand on 26/4/17.
  */
class ScalaLDAvisBuilder extends LDAvisBuilder {

  /**
    * Pass when you have existing SparkSession or simply leave it to run locally
    * @param spark SparkSession
    * @return LDAvisBuilder
    */
  override def withSparkSession(spark: SparkSession): LDAvisBuilder = {
    this.spark = Some(spark)
    this
  }

  /**
    * Offline - Load the stored fitted/transformed dataframe (i.e., pipeLineModel.transform(df))
    * @param path Directory to the stored dataframe
    * @return LDAvisBuilder
    */
  override def withTransformedDfPath(path: String): LDAvisBuilder = {
    this.transformedDfPath = Some(path)
    this
  }

  /**
    * Online - Pass the fitted/transformed dataframe (i.e., pipeLineModel.transform(df))
    * @param df Transformed dataframe with ML pipeline model
    * @return LDAvisBuilder
    */
  override def withTransformedDf(df: DataFrame): LDAvisBuilder  = {
    this.transformedDf = Some(df)
    this
  }

  /**
    * Offline - Load the trained LDA model
    * @param path Directory to the stored LDAModel
    * @return LDAvisBuilder
    */
  override def withLDAModelPath(path: String): LDAvisBuilder = {
    this.ldaModelPath = Some(path)
    this
  }

  /**
    * Online - Pass the trained LDAModel
    * @param model LDAModel
    * @return LDAvisBuilder
    */
  override def withLDAModel(model: LDAModel) = {
    this.ldaModel = Some(model)
    this
  }

  /**
    * Offline - Load the trained CountVectorizerModel model
    * @param path Directory to the stored CountVectorizerModel
    * @return LDAvisBuilder
    */
  override def withCVModelPath(path: String): LDAvisBuilder = {
    this.cvModelPath = Some(path)
    this
  }

  /**
    * Online - Pass the trained CountVectorizerModel
    * @param model CountVectorizerModel
    * @return LDAvisBuilder
    */
  override def withCVModel(model: CountVectorizerModel): LDAvisBuilder = {
    this.cvModel = Some(model)
    this.cvOutCol = Some(model.getOutputCol)
    this
  }

//  /**
//    * Offline -
//    * @param path Directory to the stored Dataframe
//    * @return LDAvisBuilder
//    */
//  override def withVocabDfPath(path: String): LDAvisBuilder = {
//    this.vocabDFPath = Some(path)
//    this
//  }

  /**
    * List of all the words in the corpus used to train the model.
    * @param words Array[String]
    * @return LDAvisBuilder
    */
  def withVocab(words: Array[String]): LDAvisBuilder = {
    this.vocab = words
    this
  }

//  def  withLDAPipeline(pipeline: Pipeline): LDAvisBuilder = {
//    this.ldaPipeline = Some(pipeline)
//    this
//  }

  /**
    * Enable or Disable debug
    * @param flag Booelan false -> Disable, true -> Enable
    * @return
    */
  def withEnableDebug(flag: Boolean = false): LDAvisBuilder = {
    this.debugFlag = flag
    this
  }

  /**
    * Builds the LDAvis with user provided data.
    * @return LDAvis.  Use LDAvis.prepareLDAVisData(...) to prepare the data for visualization
    */
  override def build: LDAvis = new ScalaLDAvis(this)
}

class ScalaLDAvis(builder: ScalaLDAvisBuilder) extends LDAvis {

  val spark: SparkSession = builder.spark.getOrElse(SparkSession
    .builder()
    .appName("SparkLDAvis")
    .master("local[4]")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .getOrCreate())

  import spark.implicits._
  val sc :SparkContext = spark.sparkContext
  sc.setLogLevel("ERROR")

  val getSize: mutable.WrappedArray[String] => Long =  _.size
  val sizeUdf = udf(getSize)

  val transformedDfPath: String = builder.transformedDfPath.getOrElse("/your/path/to/transformedDf")
  var transformedDF: DataFrame = builder.transformedDf.getOrElse(loadDF(transformedDfPath))

  val ldaModelPath = builder.ldaModelPath.getOrElse("/your/path/to/LDAmodel")
  val ldaModel: LDAModel = builder.ldaModel.getOrElse(LocalLDAModel.load(ldaModelPath))

  val cvModelPath = builder.cvModelPath.getOrElse("/your/path/to/CVmodel")
  val cvModel: CountVectorizerModel = builder.cvModel.getOrElse(CountVectorizerModel.load(cvModelPath))

  //phi Matrix [k topics x V words]
  val wordsDim = ldaModel.topicsMatrix.numRows
  val kDim = ldaModel.topicsMatrix.numCols
  val topicsTermDist: BDM[Double] = new BDM[Double](kDim, wordsDim,
    ldaModel.topicsMatrix.transpose.toArray)

  //theta Matrix [num Docs x k topics]
  val transformedDFFiltered = transformedDF.
    withColumn("doc_size", sizeUdf(transformedDF(cvModel.getInputCol))).
    select($"doc_size", transformedDF(ldaModel.getTopicDistributionCol)).
    filter($"doc_size" > 0).cache()

  transformedDFFiltered.show()

  val docTopicDist = transformedDFFiltered.
    rdd.
    flatMap(x => x.get(1).asInstanceOf[MLVector].toDense.toArray)

  val docTopicDistMat = new BDM(topicsTermDist.rows, transformedDFFiltered.count().toInt, docTopicDist.collect()).t

  val docLengths = new BDV[Double](transformedDFFiltered.
    rdd.map(row => row(0).asInstanceOf[java.lang.Long].intValue().toDouble).
    collect())

  val vocabDf = sc.parallelize(cvModel.vocabulary.zipWithIndex).toDF("term", "termIndex")

  val vocab =
    if (builder.vocab.length > 0)
      builder.vocab
    else
      vocabDf.select("term", "termIndex").
        rdd.map(r => (r.getString(0), r.getInt(1))).
        collect().sortBy(_._2).map(_._1)

  val vocabSize = vocab.length

  def loadDF(path: String): DataFrame = {

    val featuresStructToVector = udf(
      (row: Row) => {
        val indices: mutable.WrappedArray[java.lang.Long] = row(0).asInstanceOf[mutable.WrappedArray[java.lang.Long]]
        val size: Int = row(1).asInstanceOf[java.lang.Long].intValue()
        val typ: Long = row(2).asInstanceOf[Long]
        val values: mutable.WrappedArray[Double] = row(3).asInstanceOf[mutable.WrappedArray[Double]]
        new org.apache.spark.ml.linalg.SparseVector(size, indices.toArray.map(_.intValue()), values.toArray)
      }
    )

    val topicDistToVector = udf((row: Row) => {
      val typ: Long = row(0).asInstanceOf[Long]
      val values = row(1).asInstanceOf[mutable.WrappedArray[Double]].toArray
      new org.apache.spark.ml.linalg.DenseVector(values)
    })

    spark.read.json(path).
      withColumn("features", featuresStructToVector($"features")).
      withColumn("topicDistribution", topicDistToVector($"topicDistribution"))
  }

  /**
    * Transforms the topic model distributions and related corpus data into
    * the data structures needed for the visualization.
    * @param path Directory to store the visulaization data
    * @param lambdaStep Determines the interstep distance in the grid of lambda values over
                        which to iterate when computing relevance.
                        Default is 0.01. Recommended to be between 0.01 and 0.1.
    * @param plotOpts Dictionary of plotting options, right now only used for the axis labels.
    * @param R The number of terms to display in the barcharts of the visualization.
    *          Default is 30. Recommended to be roughly between 10 and 50.
    *
    *
    * Notes
    * -----
    * This implements the method of `Sievert, C. and Shirley, K. (2014):
    * LDAvis: A Method for Visualizing and Interpreting Topics, ACL Workshop on
    * Interactive Language Learning, Visualization, and Interfaces.`
    *
    * http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    */
  def prepareLDAVisData(path: String, lambdaStep: Double = 0.01,
                        plotOpts: Map[String, String] = Map("xlab" -> "PC1", "ylab" -> "PC2"),
                        R:Int =30) = {

    //[num Docs x k topics] => [k topics x num Docs] * [num Docs] => [num Docs x k topics]
    val topicFreq: BDV[Double] =
      sum(LDAvisMath.matVecElementWiseOp(docTopicDistMat.t, docLengths, (x,y) => x * y).t,
        Axis._0).inner

    if(builder.debugFlag) println("docTopicDistMat\n", docTopicDistMat)
    if(builder.debugFlag) println("docLengths\n",docLengths)
    if(builder.debugFlag) println("topicFreq\n", topicFreq)

    val topicFreqSum = sum(topicFreq)

    if(builder.debugFlag) println("topicFreqSum\n", topicFreqSum)

    //Values sorted with their zipped index
    val topicProportion = topicFreq.map(_/topicFreqSum).toArray.zipWithIndex.sortBy(_._1).reverse

    val topicOrder:IndexedSeq[Int] = topicProportion.map(_._2).toIndexedSeq

    //slicing the vector/matrix based on indexed seq
    val topicFreqSorted       = topicFreq(topicOrder).toDenseVector
    val topicTermDistsSorted =  topicsTermDist(topicOrder, ::).toDenseMatrix

    if(builder.debugFlag) println("topicFreqSorted\n", topicFreqSorted)
    if(builder.debugFlag) println("topicTermDistsSorted\n", topicTermDistsSorted)

    val termTopicFreq = LDAvisMath.matVecElementWiseOp(topicTermDistsSorted.t,
      topicFreqSorted,
      (x,y) => x * y).t

    val termFrequency: BDV[Double] = sum(termTopicFreq, Axis._0).inner

    //topicProportion is rounded to 6 decimel point
    //http://docs.oracle.com/javase/1.5.0/docs/api/java/math/RoundingMode.html#HALF_UP
    val (topicInfo, curatedTermIndex) = getTopicInfo(topicTermDistsSorted,
      new BDV(topicProportion
        .map(_._1)
        //.map(BigDecimal(_).setScale(6, BigDecimal.RoundingMode.HALF_UP).toDouble)
      ),
      termFrequency, termTopicFreq, vocab, topicOrder.toArray)

    val tokenTable = getTokenTable(curatedTermIndex.distinct.sorted,
      termTopicFreq, vocab, termFrequency).drop($"TermId")

    val topicCoordinates: Dataset[TopicCoordinates] = getTopicCoordinates(topicTermDist = topicTermDistsSorted,
      topicProportion = topicProportion)

    if(builder.debugFlag)  topicCoordinates.show()

    val clientTopicOrder = topicOrder.map(_+1).toArray

    PreparedData(topicCoordinates, topicInfo, tokenTable,
      R, lambdaStep, plotOpts, clientTopicOrder).exportTo(path)

  }

  /**
    *
    * @param topicTermDists
    * @param topicProportion
    * @param termFrequency
    * @param termTopicFreq
    * @param vocab
    * @param lambdaStep
    * @param R
    * @param nJobs
    */
  def getTopicInfo(topicTermDists: BDM[Double], topicProportion: BDV[Double], termFrequency: BDV[Double],
                   termTopicFreq: BDM[Double], vocab: Array[String], topicOrder: Array[Int],
                   lambdaStep: Double = 0.01, R: Int = 30, nJobs: Int = 0) = {

    val termPropotionSum = sum(termFrequency)

    //marginal distribution over terms (width of blue bars)
    val termPropotion: BDV[Double] = termFrequency.map(_ / termPropotionSum)

    // compute the distinctiveness and saliency of the terms:
    // this determines the R terms that are displayed when no topic is selected
    val topicTermDistsSum = sum(topicTermDists, Axis._0).inner

    val topicGivenTerm: BDM[Double] = LDAvisMath.matVecElementWiseOp(topicTermDists,
      topicTermDistsSum, (x,y) => x/y)

    val in = log(LDAvisMath.matVecElementWiseOp(topicGivenTerm.t, topicProportion, (x,y) => x/y).t)

    val kernel = (topicGivenTerm :* in)

    val distinctiveness = sum(kernel, Axis._0)

    //http://vis.stanford.edu/files/2012-Termite-AVI.pdf
    val saliency: BV[Double] = termPropotion :* distinctiveness.inner
    import org.apache.spark.sql.types.IntegerType

    val zippedTermData = ZippedDefaultTermInfo(saliency.toArray, vocab, termFrequency.toArray,
      termFrequency.toArray, Array.fill(vocab.length)("Default"))

    val defaultTermInfo_ = sc.parallelize(zippedTermData.toDefaultTermInfoArray(R)).toDS()
      .sort($"Saliency".desc)
      .limit(R)

    // Add index now...
    //https://stackoverflow.com/questions/40508489/spark-add-dataframe-column-to-another-dataframe-merge-two-dataframes
    val df1WithIndex = Utils.addColumnIndex(defaultTermInfo_.toDF())
    val df2WithIndex = Utils.addColumnIndex(spark.range(R, 0, -1).toDF("loglift"))
    val df3WithIndex = Utils.addColumnIndex(spark.range(R, 0, -1).toDF("logprob"))

    // Now time to join ... TODO any better way?
    var defaultTermInfo = df1WithIndex
      .join(df2WithIndex , Seq("columnindex"))
      .join(df3WithIndex , Seq("columnindex"))
      .drop("columnindex")
      .sort($"Saliency".desc)
      .drop($"Saliency")
      .toDF()

    val defaultTermIndex = saliency.toArray.zipWithIndex.sortWith((x,y) => x._1 > y._1)
      .map(_._2)
      .take(R)

    val logLift: BDM[Double] = log(LDAvisMath.matVecElementWiseOp(topicTermDists, termPropotion, (x,y) => x/y))

    val logTtd = log(topicTermDists)

    val lambdaSeq: Array[Double] = BigDecimal("0.00") to BigDecimal("1.0") by BigDecimal(lambdaStep) map (_.toDouble) toArray

    val topTerms: BDM[Int] = Utils.findRelevanceChunks(logLift, logTtd, R, lambdaSeq)

    val topicOrderTuple = topicOrder.zipWithIndex

    def getTopicDF(topicOrderTuple: Array[(Int, Int)]/*old, new*/,
                   topTerms: BDM[Int]): (IndexedSeq[DataFrame], Array[Int]) = {
      assert(topicOrderTuple.length == topTerms.rows)

      var curatedTermIndex: Array[Int] = Array()

      val lisOfDfs = (0 until topTerms.rows).map { case currentRowIndex =>
        val (originalID, newTopicId) = topicOrderTuple(currentRowIndex)
        val termIndex: IndexedSeq[Int] = topTerms(currentRowIndex, ::).inner.toArray.distinct.toIndexedSeq

        curatedTermIndex = curatedTermIndex ++: termIndex.toArray

        val rows = ZippedTopicTopTermRows(
          Term = termIndex.map(vocab(_)).toArray,
          Freq = Utils.matrixtoLoc(termTopicFreq, topicOrder, originalID, termIndex).toArray,
          Total = termFrequency(termIndex).toArray,
          logprob = Utils.matrixtoLoc(logTtd, topicOrder, originalID, termIndex).toArray,
          loglift = Utils.matrixtoLoc(logLift, topicOrder, originalID, termIndex).toArray,
          Category = Array.fill(termIndex.length)("Topic"+(newTopicId+1))
        ).toTopicTopTermRow()

        sc.parallelize(rows).toDF()
      }

      (lisOfDfs, curatedTermIndex)
    }

    val (lisOfDfs: IndexedSeq[DataFrame], curatedTermIndex: Array[Int]) = getTopicDF(topicOrderTuple, topTerms.t)

    lisOfDfs.foreach(df => defaultTermInfo = defaultTermInfo.union(df))

    (defaultTermInfo, defaultTermIndex ++: curatedTermIndex)

  }


  def getTokenTable(termIndex: Array[Int], termTopicFreq: BDM[Double],
                    vocab: Array[String], termFrequency: BDV[Double]) = {

    val topTopicTermsFreq = termTopicFreq(::, termIndex.toIndexedSeq).toDenseMatrix

    val K = termTopicFreq.rows

    //TODO optimize here
    def unstack(matrix: BDM[Double]): Array[TokenTable] = {
      matrix.mapPairs{
        case ((row, col), value) =>
          var option: Option[TokenTable] =
            if(value >= 0.5) {
              Some(TokenTable(TermId= termIndex(col), Topic = row+1,
                Freq = value/termFrequency(col),
                Term = vocab(termIndex(col))))
            } else  {
              None //Filter
            }
          option
      }.toArray.filter(_.isDefined).map(_.get) //TODO
    }

    sc.parallelize(unstack(topTopicTermsFreq)).toDS().sort($"Term", $"Topic")
  }

  def getTopicCoordinates(topicTermDist: BDM[Double], topicProportion: Array[(Double, Int)]): Dataset[TopicCoordinates] = {

    val K = topicTermDist.rows

    val mdsRes: BDM[Double]  = multiDimensionScaling(topicTermDist)

    if(builder.debugFlag) println("mdsRes: ", mdsRes)

    assert(mdsRes.rows == K)
    assert(mdsRes.cols == 2)

    //Map the [K x 2] matrix to Dataframe Row with case class TopicCoordinates
    val dfData = (0 until K).map(i => TopicCoordinates(mdsRes(i,0), mdsRes(i, 1), i+1, 1, topicProportion(i)._1 * 100))

    sc.parallelize(dfData).toDS()
  }

  /**
    * A function that takes `topicTermDist` as an input and outputs a
    *  `n_topics` by `2`  distance matrix. The output approximates the distance
    *  between topics.
    * @param topicTermDist
    * @return A Matrix of shape [k topics x 2]
    */
  private def multiDimensionScaling(topicTermDist: BDM[Double]): BDM[Double]  = {

    val pairDist = LDAvisMath.pairNDimDistance(topicTermDist)

    val distMatrix = LDAvisMath.squareForm(pairDist)

    LDAvisMath.PCoA(distMatrix)
  }

}

