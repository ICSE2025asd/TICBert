import pymysql


def get_connection(sql_config: dict, database=None, charset="utf8mb4", cursorClass=pymysql.cursors.SSCursor):
    if database is None:
        connection = pymysql.connect(host=sql_config["host"], user=sql_config["user"],
                                     password=sql_config["password"], cursorclass=cursorClass,
                                     autocommit=sql_config["autocommit"])
    else:
        connection = pymysql.connect(host=sql_config["host"], user=sql_config["user"],
                                     password=sql_config["password"], database=database,
                                     charset=charset, cursorclass=cursorClass, autocommit=sql_config["autocommit"])
    return connection


def get_issue_component(sql_config: dict, database: str, project: str):
    connection = get_connection(sql_config, database=database, cursorClass=pymysql.cursors.SSDictCursor)
    result = []
    try:
        with connection.cursor() as cursor:
            sql = f"SELECT * FROM issue JOIN issue_component on issue.issue_key=issue_component.issue_key WHERE issue.issue_key LIKE \"{project}%\""
            cursor.execute(sql)
            result = cursor.fetchall()
    except Exception as e:
        print(e)
    finally:
        connection.close()
    return result


def get_component_count(sql_config: dict, database: str, project: str, limit=1000):
    connection = get_connection(sql_config, database=database, cursorClass=pymysql.cursors.SSDictCursor)
    result = []
    try:
        with connection.cursor() as cursor:
            sql = f"SELECT component, COUNT(*) as count FROM issue_component WHERE issue_key LIKE \"{project}%\" GROUP BY component ORDER BY count DESC LIMIT {limit}"
            cursor.execute(sql)
            result = cursor.fetchall()
    except Exception as e:
        print(e)
    finally:
        connection.close()
    return result


def get_component_by_time(sql_config: dict, database: str, project: str):
    connection = get_connection(sql_config, database=database, cursorClass=pymysql.cursors.SSDictCursor)
    result = []
    try:
        with connection.cursor() as cursor:
            sql = f"SELECT issue_component.component, MIN(issue.create_time) AS first_occur FROM " \
                  f"issue JOIN issue_component on issue.issue_key=issue_component.issue_key WHERE " \
                  f"issue.issue_key LIKE \"{project}%\" GROUP BY issue_component.component " \
                  f"ORDER BY first_occur ASC"
            cursor.execute(sql)
            result = cursor.fetchall()
    except Exception as e:
        print(e)
    finally:
        connection.close()
    return result


def get_developer_component_count(sql_config: dict, database: str, project: str, developer: str):
    connection = get_connection(sql_config, database=database, cursorClass=pymysql.cursors.SSDictCursor)
    result = []
    try:
        with connection.cursor() as cursor:
            sql = f"SELECT component, {developer}, COUNT(*) as cnt FROM(" \
                  f"SELECT issue.issue_key, creator, component FROM issue " \
                  f"JOIN issue_component on issue.issue_key=issue_component.issue_key " \
                  f"WHERE issue.issue_key LIKE \"{project}%\") as t1 " \
                  f"GROUP BY component, {developer}"
            cursor.execute(sql)
            result = cursor.fetchall()
    except Exception as e:
        print(e)
    finally:
        connection.close()
    return result
