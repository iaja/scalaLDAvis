import Build._

unmanagedSourceDirectories in Compile += baseDirectory.value / "examples" / "src"
unmanagedResourceDirectories in Compile += baseDirectory.value / "resources"
unmanagedResources in Compile += baseDirectory.value / "resources"
unmanagedResources in Test += baseDirectory.value / "resources"
unmanagedResources in Runtime += baseDirectory.value / "resources"


lazy val clusterRoot = Project(id = "scalaLDAVis", base = file("."))
  .settings(BuildSettings.clusterSettings)

lazy val localRoot = Project(id = "scalaLDAVis-local", base = file("."))
  .settings(BuildSettings.intelliJSettings)


//Sonatype Publishing Settings

val username = "iaja"
val repo = "scalaLDAvis"

publishTo := Some(if (isSnapshot.value) Opts.resolver.sonatypeSnapshots else Opts.resolver.sonatypeStaging)

sonatypeProfileName := "com.github.iaja"

publishMavenStyle := true

homepage := Some(url(s"https://github.com/$username/$repo"))
licenses += "Apache-2.0" -> url(s"https://github.com/$username/$repo/blob/master/LICENSE")
scmInfo := Some(ScmInfo(url(s"https://github.com/$username/$repo"),
  s"scm:git@github.com:$username/$repo.git"))
//apiURL := Some(url(s"https://$username.github.io/$repo/latest/api/"))
//releaseCrossBuild := true

releasePublishArtifactsAction := PgpKeys.publishSigned.value


publishArtifact in Test := false



credentials ++= (for {
  username <- sys.env.get("SONATYPE_USERNAME")
  password <- sys.env.get("SONATYPE_PASSWORD")
} yield Credentials("Sonatype Nexus Repository Manager", "oss.sonatype.org", username, password)).toSeq

pomIncludeRepository := { _ => false }

import ReleaseTransformations._

releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  ReleaseStep(action = Command.process("publishSigned", _), enableCrossBuild = true),
  setNextVersion,
  commitNextVersion,
  ReleaseStep(action = Command.process("sonatypeReleaseAll", _), enableCrossBuild = true),
  pushChanges
)

developers := List(
  Developer(
    id    = "Mageswaran1989",
    name  = "Mageswaran Dhandapani",
    email = "mageswaran1989@gmail.com",
    url   = url(s"https://github.com/$username/$repo")
  )
)


//useGpg := true
pgpReadOnly := false

credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")

//Spark Package Settings
credentials += Credentials("Spark Packages Realm",
  "spark-packages.org",
  sys.env.get("GITHUB_USERNAME").getOrElse("set your GITHUB Credentials"),
  sys.env.get("GITHUB_PERSONAL_ACCESS_TOKEN").getOrElse("set your GITHUB Credentials"))

spName := "com.github.iaja/scalaLDAvis"
spShortDescription := "A vanilla Scala port of https://github.com/bmabey/pyLDAvis using Apache Spark as backend"

spDescription :=
  """Prepares the Spark trained LDA Model to be visualised
    |in a compact 2-D graph, preserving the topics similarity.""".stripMargin

spIgnoreProvided := true
sparkVersion := Dependencies.sparkVersion

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

spIncludeMaven := true
