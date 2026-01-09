.PHONY: help build-core test-core smoke-test

help:
	hecho "MusicPrint E2E Engineering"
	hecho "========================="
	hecho "make build-core  : Compile C++ Searcher"
	hecho "make test-core   : Run C++ Unit Tests"
	hecho "make smoke-test  : Run Full E2E Test (Real Audio -> Meta)"

build-core:
	docker run --rm -v ${CURDIR}:/workspace -w /workspace/libmusicprint gcc:12 \
		bash -c "apt-get update && apt-get install -y cmake && mkdir -p build && cd build && cmake .. && make cli_search"

test-core:
	cd libmusicprint && make test

smoke-test: build-core
	mkdir -p ${CURDIR}/tmp_fixtures
	
	# 1. Generate Fixtures (Python Container)
	docker run --rm -v ${CURDIR}:/workspace -v ${CURDIR}/tmp_fixtures:/fixtures \
		-e FIXTURE_DIR=/fixtures \
		-w /workspace \
		musicprint-pipeline:latest \
		python3 /workspace/tests/smoke_test_generator.py
	
	# 2. Run Search (C++ Container)
	docker run --rm -v ${CURDIR}:/workspace -v ${CURDIR}/tmp_fixtures:/fixtures gcc:12 \
		/workspace/libmusicprint/build/cli_search \
		/fixtures/query.bin /fixtures/index.bin /fixtures/centroids.bin \
		/workspace/meta_tokenizer_pipeline/release/music_meta.bin \
		/workspace/meta_tokenizer_pipeline/release/music_decoder.bin
