package com.iaja.ldavis.examples

import com.iaja.ldavis.ScalaLDAvisBuilder

/**
  * Created by mageswarand on 27/6/17.
  *
  * !!!Attention: Run com.iaja.ldavis.DataPreparation before running this example
  */
object LDAVis {
  def main(args: Array[String]): Unit = {

    val sparkLDAvis = new ScalaLDAvisBuilder()
      .withTransformedDfPath("/tmp/scalaLDAvis/model/transformedDF") //features, topicDistribution(num_docs x K)
      .withLDAModelPath("/tmp/scalaLDAvis/model/spark-lda") //topic_term_dist [k topics x V words]
      .withCVModelPath("/tmp/scalaLDAvis/model/cv-model")
      .withEnableDebug(true)
      .build

    sparkLDAvis.prepareLDAVisData("/tmp/scalaLDAvis/scalaLDAVis")

  }
}
