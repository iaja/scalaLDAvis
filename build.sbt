import Build._

unmanagedSourceDirectories in Compile += baseDirectory.value / "examples" / "src"
unmanagedResourceDirectories in Compile += baseDirectory.value / "resources"
unmanagedResources in Compile += baseDirectory.value / "resources"
unmanagedResources in Test += baseDirectory.value / "resources"
unmanagedResources in Runtime += baseDirectory.value / "resources"


lazy val localRoot = Project(id = "scalaLDAVis", base = file("."))
                  .settings(BuildSettings.clusterSettings)

lazy val clusterRoot = Project(id = "scalaLDAVis-local", base = file("."))
  .settings(BuildSettings.intelliJSettings)


//Sonatype Publishing Settings

val username = "iaja"
val repo = "scalaLDAvis"

homepage := Some(url(s"https://github.com/$username/$repo"))
licenses += "Apache-2.0" -> url(s"https://github.com/$username/$repo/blob/master/LICENSE")
scmInfo := Some(ScmInfo(url(s"https://github.com/$username/$repo"),
                  s"git@github.com:$username/$repo.git"))
apiURL := Some(url(s"https://$username.github.io/$repo/latest/api/"))
releaseCrossBuild := true

releasePublishArtifactsAction := PgpKeys.publishSigned.value

publishMavenStyle := true
publishArtifact in Test := false

publishTo := Some(if (isSnapshot.value) Opts.resolver.sonatypeSnapshots else Opts.resolver.sonatypeStaging)

credentials ++= (for {
  username <- sys.env.get("SONATYPE_USERNAME")
  password <- sys.env.get("SONATYPE_PASSWORD")
} yield Credentials("Sonatype Nexus Repository Manager", "oss.sonatype.org", username, password)).toSeq

pomIncludeRepository := { _ => false }

import ReleaseTransformations._
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  //runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  publishArtifacts,
  setNextVersion,
  commitNextVersion,
  releaseStepCommand("sonatypeReleaseAll"),
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


useGpg := true
pgpReadOnly := false

credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")