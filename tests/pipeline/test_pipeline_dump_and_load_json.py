import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from tubular.imputers import MeanImputer, MedianImputer
from tubular.nominal import GroupRareLevelsTransformer, OneHotEncodingTransformer
from tubular.pipeline import dump_pipeline_to_json, load_pipeline_from_json


class FakeTransformer:
    jsonable = False


class TestPipelineDumpAndLoadJson:
    """Tests for dump_pipeline_to_json() and load_pipeline_from_json()."""

    def test_dump_pipeline_then_load_pipeline(self):
        df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})

        median_imputer = MedianImputer(columns=["b"])
        mean_imputer = MeanImputer(columns=["b"])

        original_pipeline = Pipeline(
            [("median_imputer", median_imputer), ("mean_imputer", mean_imputer)]
        )

        original_pipeline.fit(df, df["a"])

        pipeline_json = dump_pipeline_to_json(original_pipeline)

        pipeline = load_pipeline_from_json(pipeline_json)

        assert len(original_pipeline.steps) == len(pipeline.steps), (
            f"number of steps in the pipeline does not match with that of original pipeline, expected {len(original_pipeline.steps)} steps but got {len(pipeline.steps)}"
        )

        for i, (original_transformer, loaded_transformer) in enumerate(
            zip(original_pipeline.steps, pipeline.steps, strict=False)
        ):
            assert original_transformer[0] == loaded_transformer[0], (
                f"loaded pipeline does not match the original pipeline at step {i}, expected step name {original_transformer[0]} but got {loaded_transformer[0]}"
            )

            original_transformer_dict = original_transformer[1].__dict__
            # removing  built_from_json attr, as the two transformers are expected to differ here
            original_transformer_dict.pop("built_from_json", None)
            loaded_transformer_dict = loaded_transformer[1].__dict__
            loaded_transformer_dict.pop("built_from_json", None)
            assert original_transformer_dict == loaded_transformer_dict, (
                f"loaded pipeline does not match the original pipeline at step {i}, expected step {original_transformer_dict} but got {loaded_transformer_dict}"
            )

    def test_dump_pipeline_to_json_output(self):
        df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})

        median_imputer = MedianImputer(columns=["b"])
        mean_imputer = MeanImputer(columns=["b"])

        original_pipeline = Pipeline(
            [("median_imputer", median_imputer), ("mean_imputer", mean_imputer)]
        )

        original_pipeline.fit(df, df["a"])

        actual_json = dump_pipeline_to_json(original_pipeline)
        transformers = ["median_imputer", "mean_imputer"]
        for transformer in transformers:
            # tubular version will differ locally vs in CI, so best to drop from test
            del actual_json[transformer]["tubular_version"]
        expected_json = {
            "mean_imputer": {
                "classname": "MeanImputer",
                "fit": {
                    "is_fitted_": True,
                    "impute_values_": mean_imputer.impute_values_,
                },
                "init": {
                    "columns": mean_imputer.columns,
                    "copy": mean_imputer.copy,
                    "return_native": mean_imputer.return_native,
                    "verbose": mean_imputer.verbose,
                    "weights_column": mean_imputer.weights_column,
                },
            },
            "median_imputer": {
                "classname": "MedianImputer",
                "fit": {
                    "is_fitted_": True,
                    "impute_values_": median_imputer.impute_values_,
                },
                "init": {
                    "columns": median_imputer.columns,
                    "copy": median_imputer.copy,
                    "return_native": median_imputer.return_native,
                    "verbose": median_imputer.verbose,
                    "weights_column": median_imputer.weights_column,
                },
            },
        }

        for i, transformer in enumerate(transformers):
            assert actual_json[transformer] == expected_json[transformer], (
                f"loaded json pipeline does not match the original pipeline at step {i}, expected step {expected_json[transformer]} but got {actual_json[transformer]}"
            )

    def test_dump_transformer_not_jsonable(self):
        good_transformer = MeanImputer(columns="a")
        bad_transformer = FakeTransformer()

        original_pipeline = Pipeline(
            [("fake_transformer", bad_transformer), ("mean_imputer", good_transformer)]
        )

        with pytest.raises(
            RuntimeError,
        ):
            dump_pipeline_to_json(original_pipeline)

    def test_rare_and_one_hot_pipeline_round_trip_keeps_signals_distinct(self):
        """Trade areas grouped as rare in training and unseen trade areas at scoring
        remain distinct signals, with a fixed output column set, after json round trip."""

        train = pl.DataFrame(
            {
                "trade_area": [
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "c",
                    "d",
                    "e",
                    "a",
                    "b",
                ],
            }
        )

        pipeline = Pipeline(
            [
                (
                    "rare",
                    GroupRareLevelsTransformer(
                        columns=["trade_area"],
                        cut_off_percent=0.15,
                        unseen_level_name="unseen",
                    ),
                ),
                (
                    "one_hot",
                    OneHotEncodingTransformer(
                        columns=["trade_area"],
                        additional_categories={"trade_area": ["rare", "unseen"]},
                    ),
                ),
            ]
        )

        pipeline.fit(train)

        score = pl.DataFrame(
            {
                "trade_area": [
                    "a",  # frequent, non rare
                    "c",  # rare during training
                    "new_mall",  # unseen at training time
                ],
            }
        )

        expected_columns = [
            "trade_area",
            "trade_area_a",
            "trade_area_b",
            "trade_area_rare",
            "trade_area_unseen",
        ]

        pre_dump_output = pipeline.transform(score)

        assert pre_dump_output.columns == expected_columns

        pre_dump_rows = pre_dump_output.to_dicts()

        loaded_pipeline = load_pipeline_from_json(dump_pipeline_to_json(pipeline))

        loaded_output = loaded_pipeline.transform(score)

        assert loaded_output.columns == expected_columns

        assert loaded_output.to_dicts() == pre_dump_rows

        # the two signals are different columns with different values
        rare_col = [row["trade_area_rare"] for row in pre_dump_rows]
        unseen_col = [row["trade_area_unseen"] for row in pre_dump_rows]

        assert rare_col == [False, True, False]
        assert unseen_col == [False, False, True]
