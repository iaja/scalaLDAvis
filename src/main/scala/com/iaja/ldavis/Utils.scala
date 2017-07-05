package com.iaja.ldavis

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import spray.json._

import scala.annotation.tailrec

/**
  * Created by mageswarand on 6/6/17.
  */
object Utils {


  object DfToJson {

    //    //https://medium.com/@sinisalouc/overcoming-type-erasure-in-scala-8f2422070d20
    //    def classAccessors[T: TypeTag]: List[(String, String)] = typeOf[T].members.collect {
    //      case m: MethodSymbol if m.isCaseAccessor => { //If the pmember is  constructor paramaeter
    //        (m.name.toString, m.returnType.typeSymbol.name.toString)
    //      }
    //    }.toList
    //USe reflection to get the column names w.r.t to df/case class and use it create
    //Establish a relation between given DataFrame and the case class to determine the column names and their types
    //Method 1: Convert DF to RDD and iterate to create a case class and use it to create the zipped version of case class
    //Methos 2: Select the column by name and create a zipped case class  out of it

    def tokenTableToJson(df: DataFrame) = {

      import df.sparkSession.implicits._

      val data = ZippedTokenTable(TermId= Array(0),
      Topic = df.select("Topic").as[Int].collect(),
      Freq = df.select("Freq").as[Double].collect(),
      Term = df.select("Term").as[String].collect() )

      object MyJsonProtocol extends DefaultJsonProtocol {
        implicit val zippedTopicTopTermRowsFormat = jsonFormat4(ZippedTokenTable)
      }

      import MyJsonProtocol._
      data.toJson
    }

    def topicInfoToJson(df: DataFrame) = {

      import df.sparkSession.implicits._

      val data = ZippedTopicTopTermRows(Term=df.select("Term").as[String].collect(),
        Freq = df.select("Freq").as[Double].collect(),
        Total=df.select("Total").as[Double].collect(),
        Category =df.select("Category").as[String].collect() ,
        loglift = df.select("loglift").as[Double].collect(),
        logprob =df.select("logprob").as[Double].collect() )

      object MyJsonProtocol extends DefaultJsonProtocol {
        implicit val zippedTopicTopTermRowsFormat = jsonFormat6(ZippedTopicTopTermRows)
      }

      import MyJsonProtocol._
      data.toJson
    }

    def topicCoordinatesToJson(df: Dataset[TopicCoordinates]) = {

      import df.sparkSession.implicits._

      val data = ZippedTopicCoordinates(x=df.select("x").as[Double].collect(),
        y=df.select("y").as[Double].collect(),
        topics = df.select("topics").as[Int].collect(),
        cluster = df.select("cluster").as[Int].collect(),
        Freq = df.select("Freq").as[Double].collect()
      )

      object MyJsonProtocol extends DefaultJsonProtocol {
        implicit val zippedTopicCoordinatesFormat = jsonFormat5(ZippedTopicCoordinates)
      }

      import MyJsonProtocol._
      data.toJson
    }
  }


  /**
    * Add Column Index to dataframe
    */
  def addColumnIndex(df: DataFrame) = df.sparkSession.sqlContext.createDataFrame(
    // Add Column index
    df.rdd.zipWithIndex.map{case (row, columnindex) => Row.fromSeq(row.toSeq :+ columnindex)},
    // Create schema
    StructType(df.schema.fields :+ StructField("columnindex", LongType, false))
  )

  def matrixtoLoc(matrix: BDM[Double], topicOrder: Array[Int], originalID: Int,
                  topicIndex : IndexedSeq[Int]) : BDV[Double]= {
    val zippedMatrix: Map[Int, BDV[Double]] = (0 until matrix.rows).map(i =>
      (topicOrder(i), matrix(i, ::).inner) //GEtting row will return a Transpose
    ).toMap

    //First get for Map and second get for option
    val row: BDV[Double] = zippedMatrix.get(originalID).get //Access the DV
    row(topicIndex).toDenseVector //Select only the particular row
  }

  def findRelevance(logLift: BDM[Double], logTtd: BDM[Double], R: Int, lambda: Double): BDM[Int] = {

    val relevance = ((lambda * logTtd) + ((1 - lambda) * logLift)).t

    //https://stackoverflow.com/questions/30416142/how-to-find-five-first-maximum-indices-of-each-column-in-a-matrix
    //now we have to loop for each colum
    // prepare the matrix and get the Vector(indexes,Array[Int],Array[Int])

    //Rows -> Topic or Ordered Topic
    //Cols -> term index
    val listsOfIndexes = for (i <- Range(0, relevance.cols))
      yield relevance(::, i).toArray
        .zipWithIndex
        .sortWith((x, y) => x._1 > y._1)
        .take(R)
        .map(x => x._2)

    //finally conver to a DenseMatrix
    BDM(listsOfIndexes.map(_.toArray): _*).t
  }

  def concat(list: Array[BDM[Int]]): BDM[Int] = {

    @tailrec
    def concat_(list: Array[BDM[Int]], res: BDM[Int]): BDM[Int] = {
      if (list.size == 0)
        res
      else
        concat_(list.tail, BDM.vertcat(res,list.head))
    }

    concat_(list.tail, list.head)
  }

  def findRelevanceChunks(logLift: BDM[Double], logTtd: BDM[Double], R: Int, lambdaSeq: Array[Double]): BDM[Int] = {
    concat(lambdaSeq.map(findRelevance(logLift, logTtd, R, _)))
  }
}
