import pprint

import pandas as pd
from beartype import beartype


__all__ = ["generate_dataframe_fixture_code"]


@beartype
def generate_dataframe_fixture_code(
    df: pd.DataFrame,
    df_name: str = "df",
) -> str:
    """
    Generate a code string to create a pandas DataFrame from a given DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame to generate code for.
        df_name (str, optional): Name of the DataFrame in the code.
            Defaults to "df".

    Raises:
        ValueError: If the input is not a pandas DataFrame

    Returns:
        str: String containing python code to explicitly generate a pandas
            DataFrame from the input DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    data_dict_str = df.to_dict(orient="list")

    pp = pprint.PrettyPrinter(indent=4)
    data_dict_pretty_str = pp.pformat(data_dict_str)

    code_str = f"""import pandas as pd

{df_name} = pd.DataFrame(
    {data_dict_pretty_str}
)
"""
    return code_str
