from src.agents.graph import route_intent, route_clarification, route_api_check, app

def test_route_intent_general_chat():
    state = {"intent": "general_chat", "needs_clarification": False}
    assert route_intent(state) == "general_chat"

def test_route_intent_track_diet():
    state = {"intent": "track_diet", "needs_clarification": False}
    assert route_intent(state) == "extraction"

def test_route_clarification_needs_clarify():
    state = {"needs_clarification": True}
    assert route_clarification(state) == "clarify"

def test_route_clarification_no_clarify():
    state = {"needs_clarification": False}
    assert route_clarification(state) == "analyze"

def test_route_api_check_success():
    state = {"api_success": True, "retry_count": 0}
    assert route_api_check(state) == "rag"

def test_route_api_check_retry():
    state = {"api_success": False, "retry_count": 0}
    assert route_api_check(state) == "self_correction"

def test_route_api_check_max_retry():
    state = {"api_success": False, "retry_count": 2}
    assert route_api_check(state) == "rag"

def test_graph_compilation():
    assert app is not None
