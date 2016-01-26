#!/bin/bash
pylint --disable=star-args,no-member,duplicate-code,wrong-import-order,wrong-import-position,too-many-nested-blocks,too-many-boolean-expressions,deprecated-method barrista && \
pylint --disable=star-args,no-member,duplicate-code,wrong-import-order,wrong-import-position,too-many-nested-blocks,too-many-boolean-expressions,deprecated-method example.py && \
pylint --disable=star-args,no-member,duplicate-code,wrong-import-order,wrong-import-position,too-many-nested-blocks,too-many-boolean-expressions,deprecated-method setup.py && \
pylint --disable=star-args,no-member,duplicate-code,wrong-import-order,wrong-import-position,too-many-nested-blocks,too-many-boolean-expressions,deprecated-method tests.py
