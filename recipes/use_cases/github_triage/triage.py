import logging
import os
from typing import Optional, Tuple, Dict
import pandas as pd
import fire

from llm import run_llm_inference
from utils import fetch_repo_issues, validate_df_values
from plots import draw_all_plots
from pdf_report import create_report_pdf

logging.basicConfig(level=logging.INFO, filename='log.txt', format='%(asctime)s [%(levelname)-5.5s] %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def generate_issue_annotations(
    issues_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Get the annotations for the given issues.

    Args:
    - issues_df (pd.DataFrame): The DataFrame containing the issues.

    Returns:
    - Tuple[pd.DataFrame, Dict[str, int]]: A tuple containing the annotated issues DataFrame and the theme counts.
    """

    # pyre-fixme[6]
    def _categorize_issues(
        issues_metadata_df: pd.DataFrame,
    ) -> Tuple[pd.Series, Dict[str, int]]:
        """
        Categorize the issues.

        Args:
        - issues_metadata_df (pd.DataFrame): The DataFrame containing the issues metadata.

        Returns:
        - Tuple[pd.Series, Dict[str, int]]: A tuple containing the categorized issues and the theme counts.
        """
        minified_issues = issues_metadata_df[
            [
                "number",
                "summary",
                "possible_causes",
                "remediations",
                "component",
                "issue_type",
            ]
        ].to_dict(orient="records")
        themes_json = run_llm_inference(
            "assign_category",
            str(minified_issues),
            generation_kwargs={"temperature": 0.45, "max_tokens": 2048},
        )

        tmp = {}
        for t in themes_json["report"]:
            for num in t["related_issues"]:
                tmp[num] = tmp.get(num, []) + [t["theme"]]

        themes = issues_metadata_df.number.apply(
            lambda x: tmp.get(x, ["Miscellaneous"])
        )
        theme_count = {
            k["theme"]: len(k["related_issues"]) for k in themes_json["report"]
        }
        return themes, theme_count

    logger.info(f"Generating annotations for {len(issues_df)} issues")
    
    discussions = issues_df["discussion"].tolist()
    metadata = run_llm_inference(
        "parse_issue",
        discussions,
        generation_kwargs={"max_tokens": 2048, "temperature": 0.42},
    )

    # Handle the case where the LLM returns None instead of a generated response
    metadata_index = [
        issues_df.index[i] for i in range(len(metadata)) if metadata[i] is not None
    ]
    metadata = [m for m in metadata if m is not None]

    issues_metadata_df = issues_df.merge(
        pd.DataFrame(metadata, index=metadata_index), left_index=True, right_index=True
    )

    themes, theme_count = _categorize_issues(issues_metadata_df)
    issues_metadata_df["themes"] = themes

    return issues_metadata_df, theme_count


def generate_executive_reports(
    annotated_issues: pd.DataFrame,
    theme_counts: Dict,
    repo_name: str,
    start_date: str,
    end_date: str,
    save_folder: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate executive reports for the given issues.

    Args:
    - annotated_issues (pd.DataFrame): The DataFrame containing the annotated issues.
    - theme_counts (dict): A dictionary containing the theme counts.
    - repo_name (str): The name of the repository. Defaults to None.
    - start_date (str): The start date of the report. Defaults to None.
    - end_date (str): The end date of the report. Defaults to None.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the challenges DataFrame and the overview DataFrame.
    """
    logger.info(f"Generating high-level summaries from annotations...")
    
    report = {
        "repo_name": repo_name,
        "start_date": start_date,
        "end_date": end_date,
        "sentiment_count": annotated_issues["sentiment"].value_counts().to_dict(),
        "severity_count": annotated_issues["severity"].value_counts().to_dict(),
        "op_expertise_count": annotated_issues["op_expertise"].value_counts().to_dict(),
        "themes_count": theme_counts,
        "issues_created": annotated_issues["number"].nunique(),
        "open_discussion": len(
            annotated_issues[
                (annotated_issues.num_comments > 0) & (annotated_issues.closed == False)
            ]
        ),
        "closed_discussion": len(
            annotated_issues[
                (annotated_issues.num_comments > 0) & (annotated_issues.closed == True)
            ]
        ),
        "open_no_discussion": len(
            annotated_issues[
                (annotated_issues.num_comments == 0)
                & (annotated_issues.closed == False)
            ]
        ),
        "closed_no_discussion": len(
            annotated_issues[
                (annotated_issues.num_comments == 0) & (annotated_issues.closed == True)
            ]
        ),
    }

    report_input = str(
        annotated_issues[
            ["number", "summary", "possible_causes", "remediations"]
        ].to_dict("records")
    )
    overview = run_llm_inference(
        "get_overview", str(report_input), {"temperature": 0.45, "max_tokens": 4096}
    )
    report.update(overview)

    overview_df = {
        k: report[k]
        for k in [
            "repo_name",
            "start_date",
            "end_date",
            "issues_created",
            "open_discussion",
            "closed_discussion",
            "open_no_discussion",
            "closed_no_discussion",
        ]
    }
    overview_df["open_questions"] = [report["open_questions"]]
    overview_df["executive_summary"] = [report["executive_summary"]]

    for col in [
        "sentiment_count",
        "severity_count",
        "op_expertise_count",
        "themes_count",
    ]:
        d = report[col]
        for k, v in d.items():
            overview_df[f"{col}_{k}"] = v

    overview_df = pd.DataFrame(overview_df)
    
    logger.info(f"Identifying key-challenges faced by users...")

    challenges_df = {k: report[k] for k in ["repo_name", "start_date", "end_date"]}
    challenges_df["key_challenge"] = [
        k["key_challenge"] for k in report["issue_analysis"]
    ]
    challenges_df["affected_issues"] = [
        k["affected_issues"] for k in report["issue_analysis"]
    ]
    challenges_df["possible_causes"] = [
        k["possible_causes"] for k in report["issue_analysis"]
    ]
    challenges_df["remediations"] = [
        k["remediations"] for k in report["issue_analysis"]
    ]
    challenges_df = pd.DataFrame(challenges_df)

    return challenges_df, overview_df
   
   
def main(repo_name, start_date, end_date):
    out_folder = f'output/{repo_name}/{start_date}_{end_date}'
    os.makedirs(out_folder, exist_ok=True)
    
    # Get issues data
    issues_df = fetch_repo_issues(repo_name, start_date, end_date)
    
    # Generate annotations and metadata
    annotated_issues, theme_counts = generate_issue_annotations(issues_df)
    # Validate and save generated data
    annotated_issues = validate_df_values(annotated_issues, out_folder, 'annotated_issues')
    
    # Generate high-level analysis
    challenges, overview = generate_executive_reports(annotated_issues, theme_counts, repo_name, start_date, end_date)
    # Validate and save generated data
    challenges = validate_df_values(challenges, out_folder, 'challenges')
    overview = validate_df_values(overview, out_folder, 'overview')
    
    # Create graphs and charts
    plot_folder = out_folder + "/plots"
    os.makedirs(plot_folder, exist_ok=True)
    draw_all_plots(repo_name, plot_folder, overview)
    
    # Create PDF report
    exec_summary = overview['executive_summary'].iloc[0]
    open_qs = overview['open_questions'].iloc[0]
    key_challenges_data = challenges[['key_challenge', 'possible_causes', 'remediations', 'affected_issues']].to_dict('records')
    create_report_pdf(repo_name, start_date, end_date, key_challenges_data, exec_summary, open_qs, out_folder)
    

if __name__ == "__main__":
    fire.Fire(main)