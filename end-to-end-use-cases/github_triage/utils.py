import requests
import yaml
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

CFG = yaml.safe_load(open("config.yaml", "r"))


def fetch_github_endpoint(url):
    headers = {
        "Authorization": f"Bearer {CFG['github_token']}",
        "Content-Type": "application/json"
    }
    logger.debug(f"Requesting url: {url}")
    response = requests.get(url, headers=headers, timeout=10)
    return response


def fetch_repo_issues(repo, start_date=None, end_date=None):
    time_filter = ""
    if start_date and not end_date:
        time_filter = f"+created:>{start_date}"
    if end_date and not start_date:
        time_filter = f"+created:<{end_date}"
    if start_date and end_date:
        time_filter = f"+created:{start_date}..{end_date}"
    
    url = f"https://api.github.com/search/issues?per_page=100&sort=created&order=asc&q=repo:{repo}+is:issue{time_filter}"

    samples = []

    while True:
        response = fetch_github_endpoint(url)
        
        if response.status_code == 200:
            issues = response.json()['items']
            for issue in issues:
                if issue['body'] is None:
                    continue
                
                issue['discussion'] = issue['title'] + "\n" + issue['body']
                if issue['comments'] > 0:
                    comments_response = fetch_github_endpoint(issue['comments_url']).json()
                    comments = "\n> ".join([x['body'] for x in comments_response])
                    issue['discussion'] += "\n> " + comments
                    
                samples.append(issue)
        
            # Check if there are more pages
            if "Link" in response.headers:
                link_header = [h.split(';') for h in response.headers["Link"].split(', ')]
                link_header = [x for x in link_header if "next" in x[1]]
                if link_header:
                    url = link_header[0][0].strip().replace('<', '').replace('>','')
                else:
                    break
            else:
                break
        else:
            raise Exception(f"Fetching issues failed with Error: {response.status_code} on url {url}")
        
    rows = [{
        "repo_name": repo,
        "number": d['number'],
        "html_url": d['html_url'],
        "closed": (d['state'] == 'closed'),
        "num_comments": d['comments'],
        "created_at": d["created_at"],
        "discussion": d['discussion'],
    } for d in samples]
    
    logger.info(f"Fetched {len(samples)} issues on {repo} from {start_date} to {end_date}")
    
    return pd.DataFrame(rows)


def fetch_repo_stats(repo):
    repo_info = fetch_github_endpoint(f"https://api.github.com/repos/{repo}").json()
    
    repo_stats = {
        "Total Open Issues": repo_info['open_issues_count'],
        "Total Stars": repo_info['stargazers_count'],
        "Total Forks": repo_info['forks_count'],
    }
    
    return repo_stats


def validate_df_values(df, out_folder=None, name=None):
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
    if out_folder is not None:
        path = f"{out_folder}/{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Data saved to {path}")
    return df
