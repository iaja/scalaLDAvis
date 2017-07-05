package com.iaja.ldavis

import breeze.linalg.{eig, svd, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, _}
import breeze.numerics._
/**
  * Created by mageswarand on 26/4/17.
  *
  * Some routines are written from ground-up imitating python routines
  */
object LDAvisMath {

  def klDivergence(p1: Array[Double], p2: Array[Double]): Array[Double] = {
    assert(p1.length == p2.length)

    val res = p1.zip(p2).map{case(x: Double,y: Double) => {
      if(x >0 && y >0)
        x * Math.log(x/y) - x + y
      else if(x ==0 && y >=0)
        y
      else
        Double.PositiveInfinity
    }}
    res
  }
  /**
    * Computes the Kullback-Leibler divergence
    * S = sum(pk * log(pk / qk), axis=0)
    * @param pk
    * @param qk
    * @return
    */
  def entropy(pk: BDV[Double], qk: BDV[Double]) : Double = {
    assert(pk.length == qk.length)

    val sumP = pk.toArray.sum
    val pArray = pk.toArray.map(_/sumP)

    val sumQ = qk.toArray.sum
    val qArray = qk.toArray.map(_/sumQ)

    val vec = klDivergence(pArray, qArray)
    vec.sum
  }

  def jensenShannon(pointP:  BDV[Double], pointQ:  BDV[Double]) = {
    val meanM = (pointP + pointQ) * 0.5
    val e1 = entropy(pointP, meanM)
    val e2 =  entropy(pointQ, meanM)
    0.5 * ( e1 + e2)
  }

  def pairNDimDistance(X: BDM[Double], metric: (BDV[Double], BDV[Double]) => Double  = jensenShannon) = {
    val rows = X.rows
    val cols = X.cols

    val newDim = ((rows * (rows -1)) / 2)
    val dm = BDV.zeros[Double](newDim)

    var k = 0

    for (i <- 0 until rows -1) {
      for (j <- i + 1 until rows) {
        val p = X(i, ::).inner
        val q = X(j, ::).inner
        dm(k) = metric(p,q)
        k += 1
      }
    }
    dm
  }

  //http://study.com/academy/lesson/how-to-solve-5-choose-2.html
  def binomialCoefficient(n: Int, r: Int): BigInt = {
    if (Math.min(r, n-r) == 0 ) return 1
    (BigInt(n - r + 1) to n).product /  (BigInt(1) to r).product
  }

  //http://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
  def squareForm(X: BDV[Double]) = {

    val n = X.length
    val d = Math.ceil(Math.sqrt(n * 2)).toInt

    assert( ((d * (d - 1)) / 2) == n.toInt)

    val matrix = BDM.zeros[Double](d,d)

    for (i <- 0 until d ) {
      for (j <- 0 until d) {
        val vecIndex = (binomialCoefficient(d, 2) - (binomialCoefficient(d-i,2)) + (j - i - 1)).toInt
        if( i == j) {
          matrix(i, j) = 0.0
        }
        else {
          if(matrix(i,j) == 0) //Update only once
            {
              matrix(i, j) = X(vecIndex)
              matrix(j, i) = X(vecIndex)
            }

        }
      }
    }
    matrix
  }

  def mean(v: BV[Double]) = (v.valuesIterator.sum) / v.size

  def zeroMean(m: BDM[Double]) = {
    val copy = m.copy
    for (c <- 0 until m.cols) {
      val col = copy(::, c)
      val colMean = mean(col)
      col -= colMean
    }
    copy
  }

  def PCA(data: BDM[Double], components: Int) = {

    val d = zeroMean(data)

    println("zeroMean: " , d)
    val svd.SVD(u,s,v)  = svd(d.t)  //s = eigenvalues , v = eigenvector matrix

    println("v: ", v.rows, v.cols, v)
    val model = v(0 until components, ::) //top 'components' eigenvectors
    println("Model: ", model.rows, model.cols)
    println("Model.T: ", model.t.rows, model.t.cols)
    val filter = model.t * model
    println("filter: ", filter.rows, filter.cols)
    filter * d
  }

  def PCoA(pairDistMatrix: BDM[Double], numComponents : Int = 2): BDM[Double] = {

    // code referenced from skbio.stats.ordination.pcoa
    // https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py


    //pairwise distance matrix is assumed symmetric
    val n = pairDistMatrix.rows
    val H: BDM[Double] = BDM.eye[Double](n) - BDM.ones[Double](n, n).mapValues(_/n)

    val squaredPairDistMatrix: BDM[Double] = pairDistMatrix.mapValues(x => x * x)
    val B : BDM[Double]= H * squaredPairDistMatrix * H  mapValues(x => -x / 2)

    val eigvals = eig(B).eigenvalues //Vector
    val eigvecs = eig(B).eigenvectors //Matrix

    val ix = argsort(eigvals).reverse.take(numComponents).toIndexedSeq

    val slicedEigenVals = eigvals(ix)
    val  slicedEigenVec = eigvecs(::, ix)

    val m = slicedEigenVec.toDenseMatrix
    val v = sqrt(slicedEigenVals.toDenseVector)

    val res = matVecElementWiseOp(m, v, (x,y) => x*y)
    res

  }

  //https://stackoverflow.com/questions/14881989/using-scala-breeze-to-do-numpy-style-broadcasting/14885146#14885146
  def matVecElementWiseOp(mat: BDM[Double],
                          vec: BDV[Double],
                          op: (Double, Double)=> Double): BDM[Double] = {
    mat.mapPairs({
      case ((row, col), value) => {
        op(value, vec(col))
      }
    })
  }

}