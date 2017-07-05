import sbt.Keys._
import sbt._
import sbtassembly.AssemblyPlugin.autoImport._
import sbtassembly.PathList
import sbtsparkpackage.SparkPackagePlugin.autoImport.{spIgnoreProvided, spName, sparkVersion}

object Dependencies {

  val sparkVersion = "2.1.0"

  val log = "org.slf4j" % "slf4j-log4j12" % "1.7.10"
  val config = "com.typesafe" % "config" % "1.2.1"

  val scalatest = "org.scalatest" % "scalatest_2.11" % "3.0.1" % "test"

  val spray = "io.spray" %%  "spray-json" % "1.3.3"

  val includeME = Seq(log, config, scalatest, spray)
  val spark = Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion)

  val sparkProvided = spark.map(x => x % "provided")
}

object BuildSettings {
  val buildScalaVersion = "2.11.8"

  val commonBuildSettings = Defaults.coreDefaultSettings ++ Seq(
    name := "scalaLDAvis",
    scalaVersion := buildScalaVersion,
    organization := "com.github.iaja",

    scalaVersion := buildScalaVersion,

    resolvers += Resolver.jcenterRepo,
    resolvers += "Sonatype Releases" at "http://oss.sonatype.org/content/repositories/releases",

    assemblyMergeStrategy in assembly := {
      case PathList(xs@_*) if xs.last == "UnusedStubClass.class" => MergeStrategy.first
      case x =>
        val oldStrategy = (assemblyMergeStrategy in assembly).value
        oldStrategy(x)
    }
//
//    scalacOptions ++= Seq(
//      "-deprecation",                      // Emit warning and location for usages of deprecated APIs.
//      "-encoding", "utf-8",                // Specify character encoding used by source files.
//      "-explaintypes",                     // Explain type errors in more detail.
//      "-feature",                          // Emit warning and location for usages of features that should be imported explicitly.
//      "-language:existentials",            // Existential types (besides wildcard types) can be written and inferred
//      "-language:experimental.macros",     // Allow macro definition (besides implementation and application)
//      "-language:higherKinds",             // Allow higher-kinded types
//      "-language:implicitConversions",     // Allow definition of implicit functions called views
//      "-unchecked",                        // Enable additional warnings where generated code depends on assumptions.
//      "-Xcheckinit",                       // Wrap field accessors to throw an exception on uninitialized access.
//      "-Xfatal-warnings",                  // Fail the compilation if there are any warnings.
//      "-Xfuture",                          // Turn on future language features.
//      "-Xlint:adapted-args",               // Warn if an argument list is modified to match the receiver.
//      "-Xlint:by-name-right-associative",  // By-name parameter of right associative operator.
//      "-Xlint:delayedinit-select",         // Selecting member of DelayedInit.
//      "-Xlint:doc-detached",               // A Scaladoc comment appears to be detached from its element.
//      "-Xlint:inaccessible",               // Warn about inaccessible types in method signatures.
//      "-Xlint:infer-any",                  // Warn when a type argument is inferred to be `Any`.
//      "-Xlint:missing-interpolator",       // A string literal appears to be missing an interpolator id.
//      "-Xlint:nullary-override",           // Warn when non-nullary `def f()' overrides nullary `def f'.
//      "-Xlint:nullary-unit",               // Warn when nullary methods return Unit.
//      "-Xlint:option-implicit",            // Option.apply used implicit view.
//      "-Xlint:package-object-classes",     // Class or object defined in package object.
//      "-Xlint:poly-implicit-overload",     // Parameterized overloaded implicit methods are not visible as view bounds.
//      "-Xlint:private-shadow",             // A private field (or class parameter) shadows a superclass field.
//      "-Xlint:stars-align",                // Pattern sequence wildcard must align with sequence component.
//      "-Xlint:type-parameter-shadow",      // A local type parameter shadows a type already in scope.
//      "-Xlint:unsound-match",              // Pattern match may not be typesafe.
//      "-Yno-adapted-args",                // Do not adapt an argument list (either by inserting () or creating a tuple) to match the receiver.
//      "-Ypartial-unification",             // Enable partial unification in type constructor inference
//      "-Ywarn-dead-code",                  // Warn when dead code is identified.
//      "-Ywarn-extra-implicit",             // Warn when more than one implicit parameter section is defined.
//      "-Ywarn-inaccessible",               // Warn about inaccessible types in method signatures.
//      "-Ywarn-infer-any",                  // Warn when a type argument is inferred to be `Any`.
//      "-Ywarn-nullary-override",           // Warn when non-nullary `def f()' overrides nullary `def f'.
//      "-Ywarn-nullary-unit",               // Warn when nullary methods return Unit.
//      "-Ywarn-numeric-widen",              // Warn when numerics are widened.
//      "-Ywarn-unused:implicits",           // Warn if an implicit parameter is unused.
//      "-Ywarn-unused:imports",             // Warn if an import selector is not referenced.
//      "-Ywarn-unused:locals",              // Warn if a local definition is unused.
//      "-Ywarn-unused:params",              // Warn if a value parameter is unused.
//      "-Ywarn-unused:patvars",             // Warn if a variable bound in a pattern is unused.
//      "-Ywarn-unused:privates",            // Warn if a private member is unused.
//      "-Ywarn-value-discard"               // Warn when non-Unit expression results are unused.
//    )
  )

  val intelliJSettings = commonBuildSettings ++ Seq(
    libraryDependencies ++= Dependencies.spark ++ Dependencies.includeME,
    target := baseDirectory.value / "target-local")

  val clusterSettings = commonBuildSettings ++ Seq(libraryDependencies ++= Dependencies.sparkProvided ++ Dependencies.includeME)
}