import subprocess

def get_git_revisions_hash(commits=['HEAD']):
    return [subprocess.check_output(['git', 'rev-parse', '{}'.format(x)])
            for x in commits]