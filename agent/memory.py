from memory.store import init_db, save_session, find_similar_session

def recall(topic: str):
    result = find_similar_session(topic)
    if result:
        return {"notes": [], "report": result["report"]}
    return None

def save(topic: str, notes: list, report: str):
    from memory.store import new_session_id
    save_session(new_session_id(), topic, [], notes, report)