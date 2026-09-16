class AlertEngine:

    def generate_alert(
        self,
        analysis,
        component="HVAC System"
    ):

        status = analysis["status"]
        deviation = analysis["deviation_percent"]

        # --------------------------------
        # NORMAL
        # --------------------------------

        if status == "NORMAL":

            return {
                "severity": "NORMAL",
                "title": (
                    f"{component} Operating Normally"
                ),
                "message": (
                    "Current consumption is within "
                    "the expected operating range."
                ),
                "action": (
                    "Continue monitoring."
                )
            }

        # --------------------------------
        # UNDER EXPECTED
        # --------------------------------

        if status == "UNDER_EXPECTED":

            return {
                "severity": "INFO",
                "title": (
                    f"🟢 {component} — "
                    "Lower Than Expected"
                ),
                "message": (
                    f"Energy consumption is "
                    f"{abs(deviation):.1f}% "
                    "below the expected value."
                ),
                "action": (
                    "Monitor conditions; lower "
                    "consumption may represent savings."
                )
            }

        # --------------------------------
        # WATCH
        # --------------------------------

        if status == "WATCH":

            return {
                "severity": "WATCH",
                "title": (
                    f"⚠️ {component} — "
                    "Early Excess Consumption"
                ),
                "message": (
                    f"Energy consumption is "
                    f"{deviation:.1f}% above "
                    "the expected value."
                ),
                "action": (
                    "Continue monitoring the trend."
                )
            }

        # --------------------------------
        # WARNING
        # --------------------------------

        if status == "WARNING":

            return {
                "severity": "WARNING",
                "title": (
                    f"🟠 {component} — "
                    "Abnormal Consumption"
                ),
                "message": (
                    f"Actual consumption is "
                    f"{deviation:.1f}% above "
                    "expected behavior."
                ),
                "action": (
                    "Inspect operating conditions "
                    "and monitor for persistence."
                )
            }

        # --------------------------------
        # CRITICAL
        # --------------------------------

        if status == "CRITICAL":

            return {
                "severity": "CRITICAL",
                "title": (
                    f"🔴 {component} — "
                    "Critical Excess Consumption"
                ),
                "message": (
                    f"Actual consumption is "
                    f"{deviation:.1f}% above "
                    "expected behavior."
                ),
                "action": (
                    "Immediate equipment inspection "
                    "is recommended."
                )
            }

        # --------------------------------
        # FALLBACK
        # --------------------------------

        return {
            "severity": "NORMAL",
            "title": (
                f"{component} Operating Normally"
            ),
            "message": (
                "No abnormal condition detected."
            ),
            "action": (
                "Continue monitoring."
            )
        }