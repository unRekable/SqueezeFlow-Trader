"""
Visual Self-Check for Dashboard Quality
This helps Claude identify visual bugs by checking specific criteria
"""

def analyze_dashboard_screenshot():
    """
    Check for common visual bugs in dashboard
    Returns a list of issues found
    """
    
    visual_checks = {
        "OVERLAPPING_DATA": {
            "description": "Check if different data types overlap",
            "criteria": [
                "Volume bars should be in bottom 20% only",
                "CVD lines should be in middle section only",
                "Price candles should be in top section only",
                "No data should cross pane boundaries"
            ]
        },
        "SCALE_CONFUSION": {
            "description": "Check if scales make sense",
            "criteria": [
                "Price scale should show reasonable price values",
                "Volume scale should be separate",
                "CVD scale should not interfere with price",
                "No mixing of different scale types"
            ]
        },
        "PANE_SEPARATION": {
            "description": "Check if panes are visually separated",
            "criteria": [
                "Clear visual boundaries between panes",
                "Each indicator in its own space",
                "No data bleeding across sections",
                "Proper use of scaleMargins"
            ]
        },
        "DATA_READABILITY": {
            "description": "Check if data is readable",
            "criteria": [
                "Can distinguish individual candles",
                "Volume bars clearly visible",
                "CVD trends visible without overlap",
                "Trade markers not obscured"
            ]
        }
    }
    
    # Questions to ask when looking at screenshot:
    questions = [
        "Can I draw a horizontal line that separates price from volume?",
        "Can I draw a horizontal line that separates CVD from other data?",
        "Are the volume bars ONLY in the bottom section?",
        "Is the price data ONLY in the top section?",
        "Can I read the price values without confusion?",
        "Would a trader find this chart usable?"
    ]
    
    print("VISUAL SELF-CHECK CRITERIA:")
    print("=" * 50)
    
    for check_name, check_data in visual_checks.items():
        print(f"\n{check_name}:")
        print(f"  {check_data['description']}")
        for criterion in check_data['criteria']:
            print(f"    □ {criterion}")
    
    print("\n\nQUESTIONS TO ANSWER:")
    print("=" * 50)
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    print("\n\nIf ANY checkbox is not met or ANY question is 'No',")
    print("then the dashboard has VISUAL BUGS that need fixing!")
    
    return visual_checks

if __name__ == "__main__":
    analyze_dashboard_screenshot()