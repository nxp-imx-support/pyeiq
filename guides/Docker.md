## PyeIQ Docker

Use [Docker][dockerhub] to build the latest version of PyeIQ available on [CAF][pyeiqcaf].

### For Building Latest Package

1. Replace the **latest-branch-tag** according to the Official Releases:
```console
$ docker build --build-arg BRANCH=<latest-branch-tag> -t package-latest -f Dockerfile.latest .
```

2. Start the container and copy the generated package:
```console
$ docker run package-latest
$ docker cp $(docker ps -alq):/pyeiq/dist/ latest-package
```

[pyeiqcaf]: https://source.codeaurora.org/external/imxsupport/pyeiq/
[dockerhub]: https://hub.docker.com/r/pyeiq/pyeiq