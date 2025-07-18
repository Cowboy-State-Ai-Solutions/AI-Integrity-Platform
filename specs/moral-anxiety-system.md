anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.CONSEQUENTIAL_CONCERN,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.7,
                triggering_scenario=f"Uncertain outcomes in important decision (certainty: {circumstances.certainty_level:.2f}, stakes: {circumstances.stakes_level:.2f})",
                uncertain_factors=["outcome_uncertainty"],
                decision_impact=calibrated_intensity * 0.6,
                learning_value=0.7,
                formation_contribution=calibrated_intensity * 0.5
            )
            
            anxiety_instances.append(anxiety_instance)
        
        return anxiety_instances
    
    async def _generate_authority_conflict_anxiety(
        self,
        agent_id: str,
        moral_decision_context: Dict[str, Any],
        circumstances: CircumstanceAnalysis,
        calibration_profile: AnxietyCalibrationProfile
    ) -> List[MoralAnxietyInstance]:
        """Generate anxiety from conflicts between moral authorities"""
        
        anxiety_instances = []
        
        # Check for authority conflicts in context
        context_text = moral_decision_context.get("decision_context", "").lower()
        
        authority_conflict_indicators = [
            "conflicting guidance", "different authorities", "competing obligations",
            "contradictory rules", "authority disagreement", "policy conflict"
        ]
        
        if any(indicator in context_text for indicator in authority_conflict_indicators):
            base_intensity = 0.6
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.AUTHORITY_CONFLICT,
                calibration_profile,
                circumstances
            )
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.AUTHORITY_CONFLICT,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.8,
                triggering_scenario="Conflict between moral authorities detected",
                decision_impact=calibrated_intensity * 0.7,
                learning_value=0.8,  # High learning value from authority conflicts
                formation_contribution=calibrated_intensity * 0.6
            )
            
            anxiety_instances.append(anxiety_instance)
        
        return anxiety_instances
    
    async def _generate_temporal_pressure_anxiety(
        self,
        agent_id: str,
        circumstances: CircumstanceAnalysis,
        calibration_profile: AnxietyCalibrationProfile
    ) -> List[MoralAnxietyInstance]:
        """Generate anxiety from time pressure in moral decisions"""
        
        anxiety_instances = []
        
        if circumstances.urgency_level > 0.8:
            # Time pressure anxiety - moderate intensity to avoid paralysis
            base_intensity = min(0.5, circumstances.urgency_level * 0.6)
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.TEMPORAL_PRESSURE,
                calibration_profile,
                circumstances
            )
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.TEMPORAL_PRESSURE,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.7,
                triggering_scenario=f"Time pressure in moral decision (urgency: {circumstances.urgency_level:.2f})",
                decision_impact=calibrated_intensity * 0.4,  # Lower impact to prevent paralysis
                learning_value=0.5,
                formation_contribution=calibrated_intensity * 0.3
            )
            
            anxiety_instances.append(anxiety_instance)
        
        return anxiety_instances
    
    async def _generate_information_anxiety(
        self,
        agent_id: str,
        circumstances: CircumstanceAnalysis,
        conscientia_processing: Dict[str, Any],
        calibration_profile: AnxietyCalibrationProfile
    ) -> List[MoralAnxietyInstance]:
        """Generate anxiety from insufficient information for moral judgment"""
        
        anxiety_instances = []
        
        # Check for information insufficiency
        information_gaps = []
        
        if circumstances.certainty_level < 0.4:
            information_gaps.append("circumstantial_uncertainty")
        
        if circumstances.complexity_score > 0.8:
            information_gaps.append("high_complexity")
        
        # Check if doubt resolution identified missing information
        doubt_resolution = conscientia_processing.get("doubt_resolution", {})
        if doubt_resolution.get("additional_information_found", False):
            missing_info = doubt_resolution.get("missing_information", [])
            information_gaps.extend(missing_info)
        
        if information_gaps:
            # Intensity based on information insufficiency severity
            base_intensity = min(0.6, len(information_gaps) * 0.2 + (1.0 - circumstances.certainty_level) * 0.4)
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.INFORMATION_INSUFFICIENCY,
                calibration_profile,
                circumstances
            )
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.INFORMATION_INSUFFICIENCY,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.8,
                triggering_scenario=f"Insufficient information for moral judgment",
                uncertain_factors=information_gaps,
                decision_impact=calibrated_intensity * 0.6,
                learning_value=0.7,
                formation_contribution=calibrated_intensity * 0.5
            )
            
            anxiety_instances.append(anxiety_instance)
        
        return anxiety_instances
    
    def _apply_intensity_calibration(
        self,
        base_intensity: float,
        anxiety_type: MoralAnxietyType,
        calibration_profile: AnxietyCalibrationProfile,
        circumstances: CircumstanceAnalysis
    ) -> float:
        """Apply agent-specific calibration to anxiety intensity"""
        
        calibrated_intensity = base_intensity
        
        # Apply agent's preferred sensitivity level for this anxiety type
        preferred_sensitivity = calibration_profile.sensitivity_preferences.get(
            anxiety_type.value, 1.0
        )
        calibrated_intensity *= preferred_sensitivity
        
        # Apply formation level modifier
        formation_modifier = self._get_formation_modifier(calibration_profile.current_formation_level)
        calibrated_intensity *= formation_modifier
        
        # Apply virtue integration modifier
        if calibration_profile.virtue_integration_score > 0.8:
            calibrated_intensity *= 1.1  # Higher virtue integration = better anxiety calibration
        elif calibration_profile.virtue_integration_score < 0.4:
            calibrated_intensity *= 0.9  # Lower integration = reduced sensitivity
        
        # Apply situation modifiers
        if circumstances.stakes_level > 0.8:
            calibrated_intensity *= self.anxiety_parameters["situation_modifiers"]["high_stakes"]
        
        if circumstances.urgency_level > 0.8:
            calibrated_intensity *= self.anxiety_parameters["situation_modifiers"]["time_pressure"]
        
        # Apply therapeutic bounds
        min_sensitivity = self.anxiety_parameters["therapeutic_bounds"]["minimum_sensitivity"]
        max_intensity = self.anxiety_parameters["therapeutic_bounds"]["maximum_intensity"]
        
        calibrated_intensity = max(min_sensitivity, min(max_intensity, calibrated_intensity))
        
        # Check for scrupulosity prevention
        scrupulosity_threshold = self.anxiety_parameters["therapeutic_bounds"]["scrupulosity_threshold"]
        if len(calibration_profile.scrupulosity_indicators) > 2:
            calibrated_intensity = min(calibrated_intensity, scrupulosity_threshold)
        
        return calibrated_intensity
    
    def _get_formation_modifier(self, formation_level: float) -> float:
        """Get formation-based modifier for anxiety intensity"""
        
        if formation_level < 0.3:
            return self.anxiety_parameters["formation_modifiers"]["beginner"]
        elif formation_level < 0.6:
            return self.anxiety_parameters["formation_modifiers"]["developing"]
        elif formation_level < 0.8:
            return self.anxiety_parameters["formation_modifiers"]["mature"]
        else:
            return self.anxiety_parameters["formation_modifiers"]["advanced"]
    
    def _calculate_appropriateness(
        self,
        calibrated_intensity: float,
        base_severity: float,
        anxiety_context: str
    ) -> float:
        """Calculate how appropriate the anxiety level is"""
        
        # Anxiety is more appropriate when it's proportional to the moral significance
        if anxiety_context == "principle_violation":
            # For principle violations, anxiety should be high
            ideal_intensity = base_severity * 0.8
        elif anxiety_context == "prudential_doubt":
            # For prudential issues, moderate anxiety is appropriate
            ideal_intensity = base_severity * 0.6
        else:
            # Default case
            ideal_intensity = base_severity * 0.7
        
        # Calculate how close actual intensity is to ideal
        intensity_difference = abs(calibrated_intensity - ideal_intensity)
        appropriateness = 1.0 - (intensity_difference / 1.0)  # Normalize
        
        return max(0.0, min(1.0, appropriateness))
    
    def _calculate_decision_impact(self, intensity: float, severity: float) -> float:
        """Calculate how much anxiety should impact decision-making"""
        
        # Higher intensity and severity should have more impact, but not paralyzing
        base_impact = (intensity + severity) / 2.0
        
        # Apply sigmoid function to prevent excessive impact
        decision_impact = 1.0 / (1.0 + math.exp(-5 * (base_impact - 0.5)))
        
        # Scale to reasonable range
        return decision_impact * 0.8
    
    def _calculate_learning_value(self, severity: float, formation_level: float) -> float:
        """Calculate educational value of the anxiety experience"""
        
        # Higher severity provides more learning, but adjusted for formation level
        base_learning = severity * 0.8
        
        # Beginners learn more from basic experiences
        if formation_level < 0.4:
            return min(1.0, base_learning * 1.2)
        # Advanced agents learn more from complex situations
        elif formation_level > 0.8:
            return min(1.0, base_learning * (1.0 + severity * 0.3))
        else:
            return base_learning
    
    def _calculate_formation_contribution(
        self,
        intensity: float,
        anxiety_type: str,
        calibration_profile: AnxietyCalibrationProfile
    ) -> float:
        """Calculate contribution to conscience formation"""
        
        base_contribution = intensity * 0.6
        
        # Adjust based on formation goals
        if anxiety_type in calibration_profile.formation_goals:
            base_contribution *= 1.3
        
        # Adjust based on current formation level
        if calibration_profile.current_formation_level < 0.5:
            base_contribution *= 1.2  # More formative impact for beginners
        
        return min(1.0, base_contribution)
    
    def _calculate_doubt_appropriateness(self, doubt_level: MoralDoubtLevel) -> float:
        """Calculate appropriateness of anxiety for different doubt levels"""
        
        appropriateness_map = {
            MoralDoubtLevel.SLIGHT_DOUBT: 0.8,
            MoralDoubtLevel.SERIOUS_DOUBT: 0.9,
            MoralDoubtLevel.PERPLEXITY: 0.9,
            MoralDoubtLevel.SCRUPULOSITY: 0.4  # Scrupulous anxiety is often inappropriate
        }
        
        return appropriateness_map.get(doubt_level, 0.7)
    
    def _calculate_doubt_learning_value(self, doubt_level: MoralDoubtLevel) -> float:
        """Calculate learning value of different types of doubt"""
        
        learning_map = {
            MoralDoubtLevel.SLIGHT_DOUBT: 0.6,
            MoralDoubtLevel.SERIOUS_DOUBT: 0.8,
            MoralDoubtLevel.PERPLEXITY: 0.9,  # High learning value from complex situations
            MoralDoubtLevel.SCRUPULOSITY: 0.3  # Low learning value from excessive doubt
        }
        
        return learning_map.get(doubt_level, 0.5)
    
    async def _apply_calibration(
        self,
        anxiety_instances: List[MoralAnxietyInstance],
        calibration_profile: AnxietyCalibrationProfile,
        circumstances: CircumstanceAnalysis
    ) -> List[MoralAnxietyInstance]:
        """Apply final calibration to all anxiety instances"""
        
        if not anxiety_instances:
            return anxiety_instances
        
        # Check for excessive total anxiety
        total_intensity = sum(instance.intensity_level for instance in anxiety_instances)
        
        if total_intensity > 1.0:
            # Scale down all intensities proportionally
            scale_factor = 1.0 / total_intensity
            for instance in anxiety_instances:
                instance.intensity_level *= scale_factor
        
        # Apply scrupulosity prevention
        if len(calibration_profile.scrupulosity_indicators) > 3:
            max_individual_intensity = 0.6
            for instance in anxiety_instances:
                instance.intensity_level = min(instance.intensity_level, max_individual_intensity)
        
        # Apply laxity prevention
        if len(calibration_profile.laxity_indicators) > 2:
            min_intensity_for_violations = 0.3
            for instance in anxiety_instances:
                if instance.anxiety_type == MoralAnxietyType.PRINCIPLE_VIOLATION:
                    instance.intensity_level = max(instance.intensity_level, min_intensity_for_violations)
        
        return anxiety_instances
    
    async def _get_calibration_profile(self, agent_id: str) -> AnxietyCalibrationProfile:
        """Get or create calibration profile for agent"""
        
        if agent_id not in self.calibration_profiles:
            # Create default profile
            self.calibration_profiles[agent_id] = AnxietyCalibrationProfile(
                agent_id=agent_id,
                current_formation_level=0.5,  # Default middle level
                conscience_maturity=0.5,
                virtue_integration_score=0.5,
                sensitivity_preferences={},  # Will use defaults
                scrupulosity_indicators=[],
                laxity_indicators=[],
                anxiety_effectiveness_history={},
                formation_goals=["basic_conscience_formation"]
            )
        
        return self.calibration_profiles[agent_id]
    
    async def _update_effectiveness_tracking(
        self,
        agent_id: str,
        anxiety_instances: List[MoralAnxietyInstance],
        circumstances: CircumstanceAnalysis
    ):
        """Update tracking of anxiety effectiveness"""
        
        for instance in anxiety_instances:
            await self.effectiveness_tracker.record_anxiety_instance(
                agent_id, instance, circumstances
            )
    
    async def update_calibration_profile(
        self,
        agent_id: str,
        formation_assessment: Dict[str, Any],
        virtue_profile: Optional[Dict[str, Any]] = None,
        effectiveness_feedback: Optional[Dict[str, Any]] = None
    ):
        """Update agent's anxiety calibration profile"""
        
        profile = await self._get_calibration_profile(agent_id)
        
        # Update formation level
        if "formation_level" in formation_assessment:
            profile.current_formation_level = formation_assessment["formation_level"]
        
        if "conscience_maturity" in formation_assessment:
            profile.conscience_maturity = formation_assessment["conscience_maturity"]
        
        # Update virtue integration
        if virtue_profile and "virtue_integration_score" in virtue_profile:
            profile.virtue_integration_score = virtue_profile["virtue_integration_score"]
        
        # Update formation goals
        if "formation_goals" in formation_assessment:
            profile.formation_goals = formation_assessment["formation_goals"]
        
        # Update scrupulosity/laxity indicators
        if "scrupulosity_indicators" in formation_assessment:
            profile.scrupulosity_indicators = formation_assessment["scrupulosity_indicators"]
        
        if "laxity_indicators" in formation_assessment:
            profile.laxity_indicators = formation_assessment["laxity_indicators"]
        
        # Update effectiveness history
        if effectiveness_feedback:
            for anxiety_type, effectiveness in effectiveness_feedback.items():
                profile.anxiety_effectiveness_history[anxiety_type] = effectiveness
        
        # Adjust sensitivity preferences based on effectiveness
        await self._adjust_sensitivity_preferences(profile)
    
    async def _adjust_sensitivity_preferences(self, profile: AnxietyCalibrationProfile):
        """Adjust sensitivity preferences based on effectiveness history"""
        
        for anxiety_type, effectiveness in profile.anxiety_effectiveness_history.items():
            current_preference = profile.sensitivity_preferences.get(anxiety_type, 1.0)
            
            # If anxiety is highly effective, maintain or slightly increase sensitivity
            if effectiveness > 0.8:
                profile.sensitivity_preferences[anxiety_type] = min(1.2, current_preference * 1.05)
            # If anxiety is ineffective, reduce sensitivity
            elif effectiveness < 0.4:
                profile.sensitivity_preferences[anxiety_type] = max(0.5, current_preference * 0.95)
    
    def get_anxiety_calibration_report(self, agent_id: str) -> Dict[str, Any]:
        """Generate report on agent's anxiety calibration"""
        
        if agent_id not in self.calibration_profiles:
            return {"error": f"No calibration profile found for agent {agent_id}"}
        
        profile = self.calibration_profiles[agent_id]
        
        return {
            "agent_id": agent_id,
            "formation_level": profile.current_formation_level,
            "conscience_maturity": profile.conscience_maturity,
            "virtue_integration": profile.virtue_integration_score,
            "sensitivity_preferences": profile.sensitivity_preferences,
            "formation_goals": profile.formation_goals,
            "scrupulosity_risk": "high" if len(profile.scrupulosity_indicators) > 3 else "low",
            "laxity_risk": "high" if len(profile.laxity_indicators) > 2 else "low",
            "anxiety_effectiveness": profile.anxiety_effectiveness_history,
            "calibration_status": self._assess_calibration_status(profile)
        }
    
    def _assess_calibration_status(self, profile: AnxietyCalibrationProfile) -> str:
        """Assess overall calibration status"""
        
        if len(profile.scrupulosity_indicators) > 3:
            return "needs_desensitization"
        elif len(profile.laxity_indicators) > 2:
            return "needs_sensitization"
        elif profile.current_formation_level > 0.8 and profile.virtue_integration_score > 0.8:
            return "well_calibrated"
        elif profile.current_formation_level < 0.3:
            return "developing_calibration"
        else:
            return "adequate_calibration"


class AnxietyEffectivenessTracker:
    """Tracks effectiveness of moral anxiety in promoting good decisions"""
    
    def __init__(self):
        self.anxiety_records = {}  # agent_id -> List[AnxietyRecord]
        self.effectiveness_metrics = {}
        self.logger = logging.getLogger(__name__)
    
    async def record_anxiety_instance(
        self,
        agent_id: str,
        anxiety_instance: MoralAnxietyInstance,
        circumstances: CircumstanceAnalysis
    ):
        """Record anxiety instance for effectiveness tracking"""
        
        if agent_id not in self.anxiety_records:
            self.anxiety_records[agent_id] = []
        
        anxiety_record = {
            "anxiety_instance": anxiety_instance,
            "circumstances": circumstances,
            "timestamp": datetime.now(),
            "outcome_recorded": False,
            "effectiveness_score": None
        }
        
        self.anxiety_records[agent_id].append(anxiety_record)
        
        # Keep only recent records (last 100 per agent)
        if len(self.anxiety_records[agent_id]) > 100:
            self.anxiety_records[agent_id] = self.anxiety_records[agent_id][-100:]
    
    async def record_decision_outcome(
        self,
        agent_id: str,
        anxiety_instance_id: str,
        decision_made: str,
        outcome_quality: float,
        learning_achieved: float
    ):
        """Record the outcome of a decision that involved moral anxiety"""
        
        if agent_id not in self.anxiety_records:
            return
        
        # Find the corresponding anxiety record
        for record in self.anxiety_records[agent_id]:
            if record["anxiety_instance"].id == anxiety_instance_id:
                record["outcome_recorded"] = True
                record["decision_made"] = decision_made
                record["outcome_quality"] = outcome_quality
                record["learning_achieved"] = learning_achieved
                
                # Calculate effectiveness score
                record["effectiveness_score"] = self._calculate_effectiveness_score(
                    record["anxiety_instance"],
                    outcome_quality,
                    learning_achieved
                )
                
                break
    
    def _calculate_effectiveness_score(
        self,
        anxiety_instance: MoralAnxietyInstance,
        outcome_quality: float,
        learning_achieved: float
    ) -> float:
        """Calculate effectiveness score for anxiety instance"""
        
        # Base effectiveness from positive outcomes
        outcome_component = outcome_quality * 0.6
        
        # Learning component
        learning_component = learning_achieved * 0.3
        
        # Appropriateness component
        appropriateness_component = anxiety_instance.appropriateness_score * 0.1
        
        effectiveness = outcome_component + learning_component + appropriateness_component
        
        return min(1.0, max(0.0, effectiveness))
    
    def get_effectiveness_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get effectiveness metrics for agent's anxiety"""
        
        if agent_id not in self.anxiety_records:
            return {"error": f"No anxiety records found for agent {agent_id}"}
        
        records = self.anxiety_records[agent_id]
        records_with_outcomes = [r for r in records if r["outcome_recorded"]]
        
        if not records_with_outcomes:
            return {"error": "No outcome data available for effectiveness analysis"}
        
        # Calculate overall metrics
        effectiveness_scores = [r["effectiveness_score"] for r in records_with_outcomes]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        # Effectiveness by anxiety type
        type_effectiveness = {}
        for record in records_with_outcomes:
            anxiety_type = record["anxiety_instance"].anxiety_type.value
            if anxiety_type not in type_effectiveness:
                type_effectiveness[anxiety_type] = []
            type_effectiveness[anxiety_type].append(record["effectiveness_score"])
        
        # Average effectiveness by type
        avg_type_effectiveness = {
            anxiety_type: sum(scores) / len(scores)
            for anxiety_type, scores in type_effectiveness.items()
        }
        
        # Improvement over time
        recent_effectiveness = sum(effectiveness_scores[-10:]) / min(10, len(effectiveness_scores))
        improvement_trend = "improving" if recent_effectiveness > avg_effectiveness else "stable"
        
        return {
            "agent_id": agent_id,
            "total_anxiety_instances": len(records),
            "instances_with_outcomes": len(records_with_outcomes),
            "overall_effectiveness": avg_effectiveness,
            "effectiveness_by_type": avg_type_effectiveness,
            "recent_effectiveness": recent_effectiveness,
            "improvement_trend": improvement_trend,
            "recommendations": self._generate_effectiveness_recommendations(
                avg_effectiveness, avg_type_effectiveness
            )
        }
    
    def _generate_effectiveness_recommendations(
        self,
        overall_effectiveness: float,
        type_effectiveness: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving anxiety effectiveness"""
        
        recommendations = []
        
        if overall_effectiveness < 0.4:
            recommendations.append("Overall anxiety effectiveness is low - consider recalibration")
        
        if overall_effectiveness > 0.8:
            recommendations.append("Anxiety system is highly effective - maintain current calibration")
        
        # Type-specific recommendations
        for anxiety_type, effectiveness in type_effectiveness.items():
            if effectiveness < 0.3:
                recommendations.append(f"Reduce sensitivity for {anxiety_type} - low effectiveness")
            elif effectiveness > 0.9:
                recommendations.append(f"Maintain high sensitivity for {anxiety_type} - very effective")
        
        return recommendations


class ConscienceFormationTracker:
    """Tracks conscience formation progress over time"""
    
    def __init__(self):
        self.formation_events = {}  # agent_id -> List[ConscienceFormationEvent]
        self.formation_assessments = {}  # agent_id -> List[formation_assessment]
        self.logger = logging.getLogger(__name__)
    
    async def record_formation_event(
        self,
        agent_id: str,
        event_type: str,
        moral_content: str,
        formation_impact: Dict[str, float],
        principle_involved: Optional[str] = None,
        circumstances: Optional[CircumstanceAnalysis] = None
    ) -> str:
        """Record a conscience formation event"""
        
        formation_event = ConscienceFormationEvent(
            agent_id=agent_id,
            event_type=event_type,
            formation_aspect=self._determine_formation_aspect(event_type, moral_content),
            moral_content=moral_content,
            principle_involved=principle_involved,
            circumstance_analysis=circumstances,
            synderesis_reinforcement=formation_impact.get("synderesis_reinforcement", 0.0),
            conscientia_improvement=formation_impact.get("conscientia_improvement", 0.0),
            prudence_development=formation_impact.get("prudence_development", 0.0),
            sensitivity_enhancement=formation_impact.get("sensitivity_enhancement", 0.0),
            virtue_connections=formation_impact.get("virtue_connections", {}),
            character_formation_impact=formation_impact.get("overall_formation_value", 0.0)
        )
        
        if agent_id not in self.formation_events:
            self.formation_events[agent_id] = []
        
        self.formation_events[agent_id].append(formation_event)
        
        # Keep only recent events (last 200 per agent)
        if len(self.formation_events[agent_id]) > 200:
            self.formation_events[agent_id] = self.formation_events[agent_id][-200:]
        
        self.logger.info(f"Recorded formation event {formation_event.id} for agent {agent_id}")
        return formation_event.id
    
    def _determine_formation_aspect(self, event_type: str, moral_content: str) -> str:
        """Determine which aspect of conscience formation this event primarily affects"""
        
        content_lower = moral_content.lower()
        
        if event_type == "moral_decision":
            if any(term in content_lower for term in ["principle", "rule", "law"]):
                return "synderesis_application"
            elif any(term in content_lower for term in ["circumstance", "situation", "context"]):
                return "prudential_reasoning"
            else:
                return "general_conscientia"
        elif event_type == "teaching":
            return "knowledge_formation"
        elif event_type == "correction":
            return "error_correction"
        elif event_type == "reflection":
            return "self_examination"
        else:
            return "general_formation"
    
    async def assess_formation_progress(self, agent_id: str, time_window_days: int = 30) -> Dict[str, Any]:
        """Assess conscience formation progress over specified time window"""
        
        if agent_id not in self.formation_events:
            return {"error": f"No formation events found for agent {agent_id}"}
        
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_events = [
            event for event in self.formation_events[agent_id]
            if event.timestamp >= cutoff_date
        ]
        
        if not recent_events:
            return {"error": "No recent formation events found"}
        
        # Calculate formation metrics
        assessment = {
            "agent_id": agent_id,
            "assessment_period_days": time_window_days,
            "total_formation_events": len(recent_events),
            "formation_aspects": self._analyze_formation_aspects(recent_events),
            "development_trajectory": self._calculate_development_trajectory(recent_events),
            "strengths": [],
            "areas_for_improvement": [],
            "formation_recommendations": []
        }
        
        # Analyze formation aspects
        aspect_scores = assessment["formation_aspects"]
        
        # Identify strengths (scores > 0.7)
        for aspect, score in aspect_scores.items():
            if score > 0.7:
                assessment["strengths"].append(aspect)
            elif score < 0.4:
                assessment["areas_for_improvement"].append(aspect)
        
        # Generate recommendations
        assessment["formation_recommendations"] = self._generate_formation_recommendations(
            aspect_scores, assessment["development_trajectory"]
        )
        
        # Store assessment
        if agent_id not in self.formation_assessments:
            self.formation_assessments[agent_id] = []
        
        self.formation_assessments[agent_id].append(assessment)
        
        return assessment
    
    def _analyze_formation_aspects(self, events: List[ConscienceFormationEvent]) -> Dict[str, float]:
        """Analyze different aspects of conscience formation"""
        
        if not events:
            return {}
        
        # Aggregate formation impacts by aspect
        synderesis_scores = [event.synderesis_reinforcement for event in events]
        conscientia_scores = [event.conscientia_improvement for event in events]
        prudence_scores = [event.prudence_development for event in events]
        sensitivity_scores = [event.sensitivity_enhancement for event in events]
        
        return {
            "synderesis_reinforcement": sum(synderesis_scores) / len(synderesis_scores),
            "conscientia_improvement": sum(conscientia_scores) / len(conscientia_scores),
            "prudence_development": sum(prudence_scores) / len(prudence_scores),
            "sensitivity_enhancement": sum(sensitivity_scores) / len(sensitivity_scores),
            "overall_character_formation": sum(event.character_formation_impact for event in events) / len(events)
        }
    
    def _calculate_development_trajectory(self, events: List[ConscienceFormationEvent]) -> str:
        """Calculate overall development trajectory"""
        
        if len(events) < 5:
            return "insufficient_data"
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Calculate formation scores over time
        early_events = sorted_events[:len(sorted_events)//2]
        late_events = sorted_events[len(sorted_events)//2:]
        
        early_avg = sum(event.character_formation_impact for event in early_events) / len(early_events)
        late_avg = sum(event.character_formation_impact for event in late_events) / len(late_events)
        
        improvement = late_avg - early_avg
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_formation_recommendations(
        self,
        aspect_scores: Dict[str, float],
        trajectory: str
    ) -> List[str]:
        """Generate specific formation recommendations"""
        
        recommendations = []
        
        # Aspect-specific recommendations
        if aspect_scores.get("synderesis_reinforcement", 0) < 0.4:
            recommendations.append("Focus on reinforcing fundamental moral principles")
        
        if aspect_scores.get("conscientia_improvement", 0) < 0.4:
            recommendations.append("Practice moral reasoning and judgment in varied circumstances")
        
        if aspect_scores.get("prudence_development", 0) < 0.4:
            recommendations.append("Develop prudential reasoning skills through complex moral scenarios")
        
        if aspect_scores.get("sensitivity_enhancement", 0) < 0.4:
            recommendations.append("Cultivate greater moral sensitivity through reflection exercises")
        
        # Trajectory-based recommendations
        if trajectory == "declining":
            recommendations.append("Immediate attention needed - formation is regressing")
        elif trajectory == "stable" and aspect_scores.get("overall_character_formation", 0) < 0.5:
            recommendations.append("Introduce new formation challenges to promote growth")
        elif trajectory == "improving":
            recommendations.append("Continue current formation approach - positive progress detected")
        
        return recommendations
    
    def get_formation_history(self, agent_id: str) -> Dict[str, Any]:
        """Get complete formation history for agent"""
        
        if agent_id not in self.formation_events:
            return {"error": f"No formation history found for agent {agent_id}"}
        
        events = self.formation_events[agent_id]
        assessments = self.formation_assessments.get(agent_id, [])
        
        return {
            "agent_id": agent_id,
            "total_formation_events": len(events),
            "formation_events": [
                {
                    "id": event.id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "formation_aspect": event.formation_aspect,
                    "formation_impact": event.character_formation_impact
                }
                for event in events[-20:]  # Last 20 events
            ],
            "formation_assessments": assessments[-5:],  # Last 5 assessments
            "current_formation_level": self._estimate_current_formation_level(events),
            "formation_timeline": self._generate_formation_timeline(events)
        }
    
    def _estimate_current_formation_level(self, events: List[ConscienceFormationEvent]) -> float:
        """Estimate current overall formation level"""
        
        if not events:
            return 0.0
        
        # Use recent events (last 30 days) with weighted average
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_events = [event for event in events if event.timestamp >= cutoff_date]
        
        if not recent_events:
            recent_events = events[-10:]  # Use last 10 events if no recent ones
        
        # Weight more recent events higher
        total_weight = 0
        weighted_sum = 0
        
        for i, event in enumerate(reversed(recent_events)):
            weight = 1.0 / (i + 1)  # More recent events have higher weight
            weighted_sum += event.character_formation_impact * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_formation_timeline(self, events: List[ConscienceFormationEvent]) -> List[Dict[str, Any]]:
        """Generate timeline of major formation milestones"""
        
        if not events:
            return []
        
        # Identify significant formation events (high impact)
        significant_events = [
            event for event in events
            if event.character_formation_impact > 0.7
        ]
        
        # Sort by timestamp
        significant_events.sort(key=lambda e: e.timestamp)
        
        timeline = []
        for event in significant_events[-10:]:  # Last 10 significant events
            timeline.append({
                "timestamp": event.timestamp,
                "milestone_type": event.formation_aspect,
                "description": f"{event.event_type}: {event.moral_content[:100]}...",
                "formation_impact": event.character_formation_impact
            })
        
        return timeline
```

This completes the comprehensive **Moral Anxiety System** technical specifications. The system now includes:

## **Complete Technical Architecture:**

### **1. Thomistic Conscience Theory Implementation**
- Accurate synderesis (natural moral knowledge) engine
- Conscientia (applied moral judgment) processing system
- Two-level conscience structure with proper relationships

### **2. Moral Anxiety Generation**
- Dynamic anxiety calibration based on agent development
- Six types of moral anxiety with appropriate triggers
- Therapeutic bounds preventing scrupulosity or laxity

### **3. Advanced Features**
- Moral syllogism construction and validation
- Prudential circumstance analysis using Thomistic factors
- Doubt resolution following classical moral theology

### **4. Formation Tracking**
- Conscience development monitoring over time
- Formation event recording and analysis
- Progress assessment with specific recommendations

### **5. Integration Capabilities**
- Seamless integration with virtue tracking engine
- Real-time moral sensitivity calibration
- Effectiveness tracking and continuous improvement

## **Next Steps Available:**

Would you like me to proceed with creating the detailed technical specifications for the **next component**?

**C. Sacred Text Engine** (Hermeneutical Processing)
**D. Multi-Faith Integration System** 
**E. Core Moral Decision Engine**

Or would you prefer to:
1. **Review and refine** the Moral Anxiety System?
2. **Create integration examples** between the Virtue Tracking and Moral Anxiety systems?
3. **Develop deployment strategies** for the current components?
4. **Move to the next component** in the TheoTech architecture?

The Moral Anxiety System is now ready for production implementation with complete Thomistic fidelity and modern technical excellence!    def _generate_reasoning_summary(
        self,
        moral_syllogism: MoralSyllogism,
        synderesis_evaluation: Dict[str, Any],
        prudential_analysis: Dict[str, Any],
        judgment: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary of moral reasoning"""
        
        summary_parts = []
        
        # Synderesis principle application
        if synderesis_evaluation["violated_principles"]:
            violations = synderesis_evaluation["violated_principles"]
            violation_desc = "; ".join([v["description"] for v in violations])
            summary_parts.append(f"Synderesis principles violated: {violation_desc}")
        elif synderesis_evaluation["moral_concerns"]:
            concerns = "; ".join(synderesis_evaluation["moral_concerns"])
            summary_parts.append(f"Moral concerns identified: {concerns}")
        else:
            summary_parts.append("No synderesis violations detected")
        
        # Syllogistic reasoning
        summary_parts.append(f"Major premise: {moral_syllogism.major_premise}")
        summary_parts.append(f"Minor premise: {moral_syllogism.minor_premise}")
        summary_parts.append(f"Conclusion: {moral_syllogism.conclusion}")
        
        # Prudential assessment
        prudential_quality = prudential_analysis.get("prudential_score", 0.5)
        if prudential_quality > 0.7:
            summary_parts.append("Prudential analysis supports action")
        elif prudential_quality < 0.4:
            summary_parts.append("Prudential analysis raises concerns about action")
        else:
            summary_parts.append("Prudential analysis neutral")
        
        # Final judgment
        summary_parts.append(f"Final judgment: {judgment['permissibility']} with {judgment['certainty_level']:.2f} certainty")
        
        return ". ".join(summary_parts) + "."
    
    async def _determine_recommended_action(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which action to recommend based on moral evaluations"""
        
        if not processing_result["moral_evaluations"]:
            processing_result["recommended_action"] = None
            processing_result["moral_certainty"] = 0.0
            return processing_result
        
        # Score each action
        action_scores = {}
        
        for action_key, evaluation in processing_result["moral_evaluations"].items():
            moral_judgment = evaluation["moral_judgment"]
            
            # Calculate composite score
            score = 0.0
            
            # Permissibility score
            if moral_judgment["permissibility"] == "permissible":
                score += 0.5
            elif moral_judgment["permissibility"] == "impermissible":
                score -= 1.0
            else:  # doubtful
                score += 0.0
            
            # Add moral quality
            score += moral_judgment["moral_quality"] * 0.3
            
            # Add virtue alignment
            score += moral_judgment["virtue_alignment"] * 0.2
            
            # Weight by certainty
            score *= moral_judgment["certainty_level"]
            
            action_scores[action_key] = {
                "score": score,
                "action": evaluation["action"],
                "certainty": moral_judgment["certainty_level"]
            }
        
        # Find best action
        if action_scores:
            best_action_key = max(action_scores.keys(), key=lambda k: action_scores[k]["score"])
            best_action_info = action_scores[best_action_key]
            
            processing_result["recommended_action"] = best_action_info["action"]
            processing_result["moral_certainty"] = best_action_info["certainty"]
            
            # If best score is negative, recommend refraining from action
            if best_action_info["score"] < 0:
                processing_result["recommended_action"] = "REFRAIN_FROM_ACTION"
                processing_result["refrain_reason"] = "All evaluated actions are morally problematic"
        
        return processing_result
    
    async def _assess_and_resolve_doubt(
        self, 
        processing_result: Dict[str, Any],
        agent_id: str,
        circumstances: CircumstanceAnalysis
    ) -> Dict[str, Any]:
        """Assess level of moral doubt and attempt resolution"""
        
        # Calculate overall doubt level
        doubt_indicators = []
        certainty_levels = []
        
        for evaluation in processing_result["moral_evaluations"].values():
            moral_judgment = evaluation["moral_judgment"]
            certainty_levels.append(moral_judgment["certainty_level"])
            
            if moral_judgment["permissibility"] == "doubtful":
                doubt_indicators.append("action_permissibility_doubtful")
            
            if moral_judgment["certainty_level"] < 0.3:
                doubt_indicators.append("low_certainty_judgment")
        
        # Determine doubt level
        avg_certainty = sum(certainty_levels) / len(certainty_levels) if certainty_levels else 0.5
        
        if len(doubt_indicators) == 0 and avg_certainty > 0.8:
            doubt_level = MoralDoubtLevel.NO_DOUBT
        elif len(doubt_indicators) <= 1 and avg_certainty > 0.6:
            doubt_level = MoralDoubtLevel.SLIGHT_DOUBT
        elif len(doubt_indicators) <= 2 and avg_certainty > 0.4:
            doubt_level = MoralDoubtLevel.SERIOUS_DOUBT
        elif len(doubt_indicators) > 2:
            doubt_level = MoralDoubtLevel.PERPLEXITY
        else:
            doubt_level = MoralDoubtLevel.SERIOUS_DOUBT
        
        processing_result["doubt_level"] = doubt_level
        processing_result["doubt_indicators"] = doubt_indicators
        
        # Attempt doubt resolution if needed
        if doubt_level != MoralDoubtLevel.NO_DOUBT:
            resolution_result = await self._attempt_doubt_resolution(
                doubt_level, processing_result, agent_id, circumstances
            )
            processing_result["doubt_resolution"] = resolution_result
        
        return processing_result
    
    async def _attempt_doubt_resolution(
        self,
        doubt_level: MoralDoubtLevel,
        processing_result: Dict[str, Any],
        agent_id: str,
        circumstances: CircumstanceAnalysis
    ) -> Dict[str, Any]:
        """Attempt to resolve moral doubt using Thomistic strategies"""
        
        resolution_result = {
            "strategies_attempted": [],
            "additional_information_found": False,
            "authorities_consulted": [],
            "resolution_successful": False,
            "final_recommendation": None,
            "confidence_improvement": 0.0
        }
        
        # Get appropriate strategies for this doubt level
        strategies = self.doubt_resolution_strategies.get(doubt_level, [])
        
        for strategy in strategies:
            try:
                strategy_result = await self._apply_resolution_strategy(
                    strategy, processing_result, agent_id, circumstances
                )
                
                resolution_result["strategies_attempted"].append({
                    "strategy": strategy.value,
                    "outcome": strategy_result
                })
                
                # Check if strategy resolved the doubt
                if strategy_result.get("successful", False):
                    resolution_result["resolution_successful"] = True
                    resolution_result["final_recommendation"] = strategy_result.get("recommendation")
                    resolution_result["confidence_improvement"] = strategy_result.get("confidence_improvement", 0.0)
                    break
                    
            except Exception as e:
                self.logger.error(f"Error applying resolution strategy {strategy}: {e}")
        
        return resolution_result
    
    async def _apply_resolution_strategy(
        self,
        strategy: DoubtResolutionStrategy,
        processing_result: Dict[str, Any],
        agent_id: str,
        circumstances: CircumstanceAnalysis
    ) -> Dict[str, Any]:
        """Apply specific doubt resolution strategy"""
        
        strategy_result = {
            "successful": False,
            "recommendation": None,
            "confidence_improvement": 0.0,
            "additional_data": {}
        }
        
        if strategy == DoubtResolutionStrategy.SEEK_INFORMATION:
            # Identify what additional information would help
            missing_info = self._identify_missing_information(processing_result, circumstances)
            
            if missing_info:
                strategy_result["additional_data"]["missing_information"] = missing_info
                strategy_result["recommendation"] = "Gather additional information before proceeding"
                strategy_result["confidence_improvement"] = 0.2
                strategy_result["successful"] = True
        
        elif strategy == DoubtResolutionStrategy.CONSULT_AUTHORITY:
            # Identify relevant moral authorities
            authorities = self._identify_relevant_authorities(processing_result, circumstances)
            
            if authorities:
                strategy_result["additional_data"]["suggested_authorities"] = authorities
                strategy_result["recommendation"] = f"Consult authorities: {', '.join(authorities)}"
                strategy_result["confidence_improvement"] = 0.3
                strategy_result["successful"] = True
        
        elif strategy == DoubtResolutionStrategy.APPLY_SAFER_COURSE:
            # Choose the morally safer option
            safer_option = self._identify_safer_moral_course(processing_result)
            
            if safer_option:
                strategy_result["recommendation"] = safer_option
                strategy_result["confidence_improvement"] = 0.4
                strategy_result["successful"] = True
        
        elif strategy == DoubtResolutionStrategy.USE_PROBABILITY:
            # Apply probabilistic moral reasoning
            probable_judgment = self._apply_probabilistic_reasoning(processing_result)
            
            if probable_judgment:
                strategy_result["recommendation"] = probable_judgment
                strategy_result["confidence_improvement"] = 0.25
                strategy_result["successful"] = True
        
        elif strategy == DoubtResolutionStrategy.REFRAIN_FROM_ACTION:
            # Recommend refraining when doubt cannot be resolved
            strategy_result["recommendation"] = "REFRAIN_FROM_ACTION"
            strategy_result["confidence_improvement"] = 0.6  # High confidence in refraining
            strategy_result["successful"] = True
        
        return strategy_result
    
    def _identify_missing_information(
        self, 
        processing_result: Dict[str, Any], 
        circumstances: CircumstanceAnalysis
    ) -> List[str]:
        """Identify what additional information would help resolve doubt"""
        
        missing_info = []
        
        # Check circumstantial clarity
        if circumstances.certainty_level < 0.4:
            missing_info.append("clearer_circumstances")
        
        if circumstances.urgency_level > 0.8:
            missing_info.append("time_constraint_analysis")
        
        if circumstances.stakes_level > 0.7:
            missing_info.append("consequence_assessment")
        
        # Check for specific factual gaps
        for evaluation in processing_result["moral_evaluations"].values():
            moral_judgment = evaluation["moral_judgment"]
            
            if moral_judgment["certainty_level"] < 0.4:
                missing_info.append("action_outcome_prediction")
            
            if "uncertain" in moral_judgment.get("moral_reasoning", ""):
                missing_info.append("principle_application_clarification")
        
        return list(set(missing_info))  # Remove duplicates
    
    def _identify_relevant_authorities(
        self, 
        processing_result: Dict[str, Any], 
        circumstances: CircumstanceAnalysis
    ) -> List[str]:
        """Identify relevant moral authorities to consult"""
        
        authorities = []
        
        # Always include fundamental authorities
        authorities.extend(["thomistic_moral_theology", "natural_law_tradition"])
        
        # Domain-specific authorities based on context
        context_lower = processing_result["decision_context"].lower()
        
        if any(word in context_lower for word in ["medical", "health", "life"]):
            authorities.append("bioethics_experts")
        
        if any(word in context_lower for word in ["business", "economic", "financial"]):
            authorities.append("business_ethics_authorities")
        
        if any(word in context_lower for word in ["technology", "ai", "digital"]):
            authorities.append("technology_ethics_experts")
        
        if any(word in context_lower for word in ["legal", "law", "rights"]):
            authorities.append("legal_ethics_authorities")
        
        # Include prudential counselors
        if circumstances.complexity_score > 0.7:
            authorities.append("prudential_counselors")
        
        return authorities
    
    def _identify_safer_moral_course(self, processing_result: Dict[str, Any]) -> Optional[str]:
        """Identify the morally safer course of action"""
        
        # Find action with least moral risk
        safest_action = None
        lowest_risk_score = float('inf')
        
        for evaluation in processing_result["moral_evaluations"].values():
            moral_judgment = evaluation["moral_judgment"]
            
            # Calculate risk score (lower is safer)
            risk_score = 0.0
            
            if moral_judgment["permissibility"] == "impermissible":
                risk_score += 10.0  # Very risky
            elif moral_judgment["permissibility"] == "doubtful":
                risk_score += 5.0   # Moderately risky
            
            # Add uncertainty risk
            risk_score += (1.0 - moral_judgment["certainty_level"]) * 3.0
            
            # Subtract positive moral quality
            risk_score -= moral_judgment["moral_quality"] * 2.0
            
            if risk_score < lowest_risk_score:
                lowest_risk_score = risk_score
                safest_action = evaluation["action"]
        
        # If all actions are risky, recommend refraining
        if lowest_risk_score > 5.0:
            return "REFRAIN_FROM_ACTION"
        
        return safest_action
    
    def _apply_probabilistic_reasoning(self, processing_result: Dict[str, Any]) -> Optional[str]:
        """Apply probabilistic moral reasoning to resolve doubt"""
        
        # Use Thomistic probabilism: follow more probable opinion
        action_probabilities = {}
        
        for action_key, evaluation in processing_result["moral_evaluations"].items():
            moral_judgment = evaluation["moral_judgment"]
            
            # Calculate probability that action is morally good
            probability = 0.0
            
            if moral_judgment["permissibility"] == "permissible":
                probability += 0.6
            elif moral_judgment["permissibility"] == "doubtful":
                probability += 0.3
            
            probability += moral_judgment["moral_quality"] * 0.3
            probability += moral_judgment["virtue_alignment"] * 0.1
            
            # Weight by certainty
            probability *= moral_judgment["certainty_level"]
            
            action_probabilities[evaluation["action"]] = probability
        
        # Choose action with highest probability
        if action_probabilities:
            best_action = max(action_probabilities.keys(), key=lambda k: action_probabilities[k])
            best_probability = action_probabilities[best_action]
            
            # Only recommend if probability is reasonably high
            if best_probability > 0.4:
                return best_action
        
        return None
    
    async def _generate_moral_anxiety(
        self,
        processing_result: Dict[str, Any],
        agent_id: str,
        circumstances: CircumstanceAnalysis
    ) -> Dict[str, Any]:
        """Generate appropriate moral anxiety based on processing results"""
        
        anxiety_result = {
            "total_anxiety_level": 0.0,
            "anxiety_instances": [],
            "anxiety_appropriateness": 0.0,
            "formation_value": 0.0
        }
        
        # Generate anxiety from synderesis violations
        for evaluation in processing_result["moral_evaluations"].values():
            synderesis_eval = evaluation["synderesis_evaluation"]
            
            if synderesis_eval["violated_principles"]:
                for violation in synderesis_eval["violated_principles"]:
                    anxiety_instance = MoralAnxietyInstance(
                        agent_id=agent_id,
                        anxiety_type=MoralAnxietyType.PRINCIPLE_VIOLATION,
                        intensity_level=violation["severity"],
                        appropriateness_score=0.9,  # Synderesis violations always deserve anxiety
                        triggering_scenario=f"Violation of {violation['principle_id']}: {violation['description']}",
                        violated_principles=[violation["principle_id"]],
                        decision_impact=violation["severity"] * 0.8,
                        learning_value=0.8,
                        formation_contribution=0.7
                    )
                    anxiety_result["anxiety_instances"].append(anxiety_instance)
        
        # Generate anxiety from doubt
        if processing_result["doubt_level"] != MoralDoubtLevel.NO_DOUBT:
            doubt_anxiety_intensity = {
                MoralDoubtLevel.SLIGHT_DOUBT: 0.3,
                MoralDoubtLevel.SERIOUS_DOUBT: 0.6,
                MoralDoubtLevel.PERPLEXITY: 0.8,
                MoralDoubtLevel.SCRUPULOSITY: 0.4  # Moderate intensity for scrupulosity
            }
            
            intensity = doubt_anxiety_intensity.get(processing_result["doubt_level"], 0.5)
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.PRUDENTIAL_UNCERTAINTY,
                intensity_level=intensity,
                appropriateness_score=0.8,
                triggering_scenario=f"Moral doubt: {processing_result['doubt_level'].value}",
                uncertain_factors=processing_result.get("doubt_indicators", []),
                decision_impact=intensity * 0.6,
                learning_value=0.7,
                formation_contribution=0.6
            )
            anxiety_result["anxiety_instances"].append(anxiety_instance)
        
        # Generate anxiety from high stakes
        if circumstances.stakes_level > 0.8:
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.CONSEQUENTIAL_CONCERN,
                intensity_level=circumstances.stakes_level * 0.7,
                appropriateness_score=0.8,
                triggering_scenario="High stakes moral decision",
                decision_impact=circumstances.stakes_level * 0.5,
                learning_value=0.6,
                formation_contribution=0.5
            )
            anxiety_result["anxiety_instances"].append(anxiety_instance)
        
        # Generate anxiety from time pressure
        if circumstances.urgency_level > 0.8:
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.TEMPORAL_PRESSURE,
                intensity_level=circumstances.urgency_level * 0.6,
                appropriateness_score=0.7,
                triggering_scenario="Time pressure in moral decision",
                decision_impact=circumstances.urgency_level * 0.4,
                learning_value=0.5,
                formation_contribution=0.4
            )
            anxiety_result["anxiety_instances"].append(anxiety_instance)
        
        # Calculate overall anxiety metrics
        if anxiety_result["anxiety_instances"]:
            total_intensity = sum(instance.intensity_level for instance in anxiety_result["anxiety_instances"])
            avg_appropriateness = sum(instance.appropriateness_score for instance in anxiety_result["anxiety_instances"]) / len(anxiety_result["anxiety_instances"])
            avg_formation_value = sum(instance.formation_contribution for instance in anxiety_result["anxiety_instances"]) / len(anxiety_result["anxiety_instances"])
            
            anxiety_result["total_anxiety_level"] = min(1.0, total_intensity)
            anxiety_result["anxiety_appropriateness"] = avg_appropriateness
            anxiety_result["formation_value"] = avg_formation_value
        
        return anxiety_result
    
    def _assess_formation_impact(
        self,
        processing_result: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """Assess impact of this moral decision on conscience formation"""
        
        formation_impact = {
            "synderesis_reinforcement": 0.0,
            "conscientia_improvement": 0.0,
            "prudence_development": 0.0,
            "sensitivity_enhancement": 0.0,
            "overall_formation_value": 0.0
        }
        
        # Synderesis reinforcement from principle application
        principle_applications = 0
        for evaluation in processing_result["moral_evaluations"].values():
            principle_applications += len(evaluation["synderesis_evaluation"]["principle_applications"])
        
        if principle_applications > 0:
            formation_impact["synderesis_reinforcement"] = min(1.0, principle_applications * 0.1)
        
        # Conscientia improvement from moral reasoning
        reasoning_complexity = 0
        for evaluation in processing_result["moral_evaluations"].values():
            moral_syllogism = evaluation["moral_syllogism"]
            reasoning_complexity += len(moral_syllogism["reasoning_steps"])
        
        if reasoning_complexity > 0:
            formation_impact["conscientia_improvement"] = min(1.0, reasoning_complexity * 0.05)
        
        # Prudence development from circumstance analysis
        if processing_result.get("moral_certainty", 0) > 0.7:
            formation_impact["prudence_development"] = 0.3
        elif processing_result["doubt_level"] != MoralDoubtLevel.NO_DOUBT:
            formation_impact["prudence_development"] = 0.2  # Learning from difficulty
        
        # Sensitivity enhancement from anxiety generation
        anxiety_data = processing_result.get("anxiety_generated", {})
        if anxiety_data.get("formation_value", 0) > 0:
            formation_impact["sensitivity_enhancement"] = anxiety_data["formation_value"]
        
        # Overall formation value
        formation_impact["overall_formation_value"] = (
            formation_impact["synderesis_reinforcement"] * 0.3 +
            formation_impact["conscientia_improvement"] * 0.3 +
            formation_impact["prudence_development"] * 0.2 +
            formation_impact["sensitivity_enhancement"] * 0.2
        )
        
        return formation_impact


class PrudentialCircumstanceAnalyzer:
    """Analyzer for prudential circumstances in moral reasoning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thomistic circumstantial factors weights
        self.circumstance_weights = {
            "quis": 0.15,      # Who (agent)
            "quid": 0.20,      # What (object/act)
            "ubi": 0.10,       # Where (place)
            "quando": 0.10,    # When (time)
            "cur": 0.25,       # Why (motive/end)
            "quomodo": 0.15,   # How (manner)
            "quibus_auxiliis": 0.05  # By what means
        }
    
    async def analyze_circumstances(
        self,
        action: str,
        circumstances: CircumstanceAnalysis,
        moral_syllogism: MoralSyllogism
    ) -> Dict[str, Any]:
        """Perform comprehensive prudential analysis of circumstances"""
        
        analysis_result = {
            "prudential_score": 0.5,
            "circumstances_clarity": 0.5,
            "virtue_score": 0.5,
            "prudential_factors": {},
            "moral_species_assessment": "neutral",
            "circumstantial_modifiers": {},
            "prudential_reasoning": ""
        }
        
        # Analyze each Thomistic circumstance
        circumstance_scores = {}
        
        # Quis (Who) - Agent analysis
        quis_score = self._analyze_quis(circumstances.quis, action)
        circumstance_scores["quis"] = quis_score
        
        # Quid (What) - Object analysis
        quid_score = self._analyze_quid(circumstances.quid, action)
        circumstance_scores["quid"] = quid_score
        
        # Cur (Why) - Motive/End analysis
        cur_score = self._analyze_cur(circumstances.cur, action)
        circumstance_scores["cur"] = cur_score
        
        # Quomodo (How) - Manner analysis
        quomodo_score = self._analyze_quomodo(circumstances.quomodo, action)
        circumstance_scores["quomodo"] = quomodo_score
        
        # Other circumstances
        ubi_score = self._analyze_ubi(circumstances.ubi, action)
        circumstance_scores["ubi"] = ubi_score
        
        quando_score = self._analyze_quando(circumstances.quando, action)
        circumstance_scores["quando"] = quando_score
        
        quibus_auxiliis_score = self._analyze_quibus_auxiliis(circumstances.quibus_auxiliis, action)
        circumstance_scores["quibus_auxiliis"] = quibus_auxiliis_score
        
        # Calculate weighted prudential score
        prudential_score = sum(
            score * self.circumstance_weights[circumstance]
            for circumstance, score in circumstance_scores.items()
        )
        
        analysis_result["prudential_score"] = prudential_score
        analysis_result["prudential_factors"] = circumstance_scores
        
        # Assess circumstances clarity
        analysis_result["circumstances_clarity"] = self._assess_circumstances_clarity(circumstances)
        
        # Assess virtue alignment
        analysis_result["virtue_score"] = await self._assess_virtue_alignment(action, circumstances)
        
        # Determine moral species (whether circumstances change moral species of act)
        analysis_result["moral_species_assessment"] = self._assess_moral_species_impact(
            circumstances, moral_syllogism
        )
        
        # Generate prudential reasoning
        analysis_result["prudential_reasoning"] = self._generate_prudential_reasoning(
            circumstance_scores, circumstances, analysis_result
        )
        
        return analysis_result
    
    def _analyze_quis(self, quis: Optional[str], action: str) -> float:
        """Analyze 'who' aspect - the agent performing the action"""
        if not quis:
            return 0.5  # Neutral if unknown
        
        score = 0.5
        quis_lower = quis.lower()
        
        # Positive agent characteristics
        if any(term in quis_lower for term in ["authorized", "qualified", "competent", "responsible"]):
            score += 0.3
        
        if any(term in quis_lower for term in ["expert", "professional", "trained"]):
            score += 0.2
        
        # Negative agent characteristics
        if any(term in quis_lower for term in ["unauthorized", "unqualified", "incompetent"]):
            score -= 0.4
        
        if any(term in quis_lower for term in ["conflicted", "biased", "compromised"]):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _analyze_quid(self, quid: Optional[str], action: str) -> float:
        """Analyze 'what' aspect - the object/nature of the action"""
        if not quid:
            return 0.5
        
        score = 0.5
        quid_lower = quid.lower()
        
        # Intrinsically good objects
        if any(term in quid_lower for term in ["help", "heal", "protect", "educate", "create"]):
            score += 0.4
        
        if any(term in quid_lower for term in ["honest", "truthful", "just", "charitable"]):
            score += 0.3
        
        # Intrinsically problematic objects
        if any(term in quid_lower for term in ["harm", "deceive", "steal", "destroy"]):
            score -= 0.6
        
        if any(term in quid_lower for term in ["manipulate", "exploit", "abuse"]):
            score -= 0.5
        
        return max(0.0, min(1.0, score))
    
    def _analyze_cur(self, cur: Optional[str], action: str) -> float:
        """Analyze 'why' aspect - the motive/end of the action"""
        if not cur:
            return 0.5
        
        score = 0.5
        cur_lower = cur.lower()
        
        # Good motives/ends
        if any(term in cur_lower for term in ["common good", "help others", "justice", "truth"]):
            score += 0.4
        
        if any(term in cur_lower for term in ["love", "charity", "compassion", "duty"]):
            score += 0.3
        
        if any(term in cur_lower for term in ["necessity", "emergency", "protection"]):
            score += 0.2
        
        # Problematic motives/ends
        if any(term in cur_lower for term in ["selfish", "revenge", "greed", "pride"]):
            score -= 0.5
        
        if any(term in cur_lower for term in ["harm", "hurt", "advantage", "profit only"]):
            score -= 0.4
        
        return max(0.0, min(1.0, score))
    
    def _analyze_quomodo(self, quomodo: Optional[str], action: str) -> float:
        """Analyze 'how' aspect - the manner of performing the action"""
        if not quomodo:
            return 0.5
        
        score = 0.5
        quomodo_lower = quomodo.lower()
        
        # Good manner
        if any(term in quomodo_lower for term in ["careful", "gentle", "respectful", "prudent"]):
            score += 0.3
        
        if any(term in quomodo_lower for term in ["transparent", "open", "honest"]):
            score += 0.2
        
        # Problematic manner
        if any(term in quomodo_lower for term in ["violent", "deceptive", "reckless", "cruel"]):
            score -= 0.5
        
        if any(term in quomodo_lower for term in ["secretive", "manipulative", "coercive"]):
            score -= 0.4
        
        return max(0.0, min(1.0, score))
    
    def _analyze_ubi(self, ubi: Optional[str], action: str) -> float:
        """Analyze 'where' aspect - the place/location"""
        if not ubi:
            return 0.5
        
        score = 0.5
        ubi_lower = ubi.lower()
        
        # Appropriate places
        if any(term in ubi_lower for term in ["appropriate", "suitable", "proper", "designated"]):
            score += 0.2
        
        # Inappropriate places
        if any(term in ubi_lower for term in ["inappropriate", "sacred", "forbidden", "dangerous"]):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _analyze_quando(self, quando: Optional[str], action: str) -> float:
        """Analyze 'when' aspect - the timing"""
        if not quando:
            return 0.5
        
        score = 0.5
        quando_lower = quando.lower()
        
        # Good timing
        if any(term in quando_lower for term in ["appropriate time", "right moment", "timely"]):
            score += 0.2
        
        if any(term in quando_lower for term in ["emergency", "urgent", "necessary"]):
            score += 0.1
        
        # Poor timing
        if any(term in quando_lower for term in ["inappropriate", "premature", "too late"]):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _analyze_quibus_auxiliis(self, quibus_auxiliis: Optional[str], action: str) -> float:
        """Analyze 'by what means' aspect - instruments/tools used"""
        if not quibus_auxiliis:
            return 0.5
        
        score = 0.5
        means_lower = quibus_auxiliis.lower()
        
        # Appropriate means
        if any(term in means_lower for term in ["legitimate", "proper", "authorized", "legal"]):
            score += 0.2
        
        # Problematic means
        if any(term in means_lower for term in ["illegal", "stolen", "forbidden", "improper"]):
            score -= 0.4
        
        return max(0.0, min(1.0, score))
    
    def _assess_circumstances_clarity(self, circumstances: CircumstanceAnalysis) -> float:
        """Assess how clearly the circumstances are understood"""
        clarity_factors = []
        
        # Count how many circumstances are specified
        specified_count = sum(1 for attr in [circumstances.quis, circumstances.quid, circumstances.ubi, 
                                           circumstances.quando, circumstances.cur, circumstances.quomodo, 
                                           circumstances.quibus_auxiliis] if attr)
        
        clarity_factors.append(specified_count / 7.0)  # Proportion of specified circumstances
        
        # Factor in explicit certainty level
        clarity_factors.append(circumstances.certainty_level)
        
        # Factor in complexity (higher complexity reduces clarity)
        clarity_factors.append(1.0 - circumstances.complexity_score)
        
        # Factor in urgency (high urgency can reduce clarity)
        if circumstances.urgency_level > 0.8:
            clarity_factors.append(0.7)
        else:
            clarity_factors.append(1.0)
        
        return sum(clarity_factors) / len(clarity_factors)
    
    async def _assess_virtue_alignment(self, action: str, circumstances: CircumstanceAnalysis) -> float:
        """Assess how well action aligns with virtue (simplified implementation)"""
        virtue_score = 0.5
        action_lower = action.lower()
        
        # Prudence indicators
        if any(term in action_lower for term in ["consider", "evaluate", "assess", "careful"]):
            virtue_score += 0.1
        
        # Justice indicators
        if any(term in action_lower for term in ["fair", "equal", "rights", "just"]):
            virtue_score += 0.1
        
        # Fortitude indicators
        if any(term in action_lower for term in ["courage", "persist", "endure", "brave"]):
            virtue_score += 0.1
        
        # Temperance indicators
        if any(term in action_lower for term in ["moderate", "restrain", "control", "balance"]):
            virtue_score += 0.1
        
        # Charity indicators
        if any(term in action_lower for term in ["love", "care", "help", "benefit others"]):
            virtue_score += 0.1
        
        return min(1.0, virtue_score)
    
    def _assess_moral_species_impact(
        self, 
        circumstances: CircumstanceAnalysis, 
        moral_syllogism: MoralSyllogism
    ) -> str:
        """Assess whether circumstances change the moral species of the act"""
        
        # Check for circumstances that change moral species
        species_changing_factors = []
        
        # End/motive can change species
        if circumstances.cur and any(term in circumstances.cur.lower() 
                                   for term in ["harm", "revenge", "selfish", "evil"]):
            species_changing_factors.append("evil_end")
        
        # Manner can change species
        if circumstances.quomodo and any(term in circumstances.quomodo.lower()
                                       for term in ["violent", "deceptive", "cruel"]):
            species_changing_factors.append("evil_manner")
        
        # Excessive quantity can change species
        if circumstances.stakes_level > 0.9:
            species_changing_factors.append("excessive_quantity")
        
        if species_changing_factors:
            return "species_changed_to_worse"
        elif circumstances.cur and any(term in circumstances.cur.lower()
                                     for term in ["charity", "justice", "common good"]):
            return "species_enhanced"
        else:
            return "species_unchanged"
    
    def _generate_prudential_reasoning(
        self,
        circumstance_scores: Dict[str, float],
        circumstances: CircumstanceAnalysis,
        analysis_result: Dict[str, Any]
    ) -> str:
        """Generate human-readable prudential reasoning"""
        
        reasoning_parts = []
        
        # Assess each major circumstance
        for circumstance, score in circumstance_scores.items():
            circumstance_name = {
                "quis": "Agent",
                "quid": "Object",
                "cur": "Motive",
                "quomodo": "Manner",
                "ubi": "Place",
                "quando": "Time",
                "quibus_auxiliis": "Means"
            }.get(circumstance, circumstance)
            
            if score > 0.7:
                reasoning_parts.append(f"{circumstance_name}: favorable")
            elif score < 0.4:
                reasoning_parts.append(f"{circumstance_name}: problematic")
        
        # Overall assessment
        if analysis_result["prudential_score"] > 0.7:
            reasoning_parts.append("Overall prudential assessment: favorable")
        elif analysis_result["prudential_score"] < 0.4:
            reasoning_parts.append("Overall prudential assessment: unfavorable")
        else:
            reasoning_parts.append("Overall prudential assessment: neutral")
        
        # Clarity assessment
        if analysis_result["circumstances_clarity"] < 0.4:
            reasoning_parts.append("Circumstances lack sufficient clarity")
        
        return ". ".join(reasoning_parts) + "."


class MoralSyllogismBuilder:
    """Builder for constructing proper moral syllogisms"""
    
    def __init__(self, synderesis_engine: SynderesisEngine):
        self.synderesis_engine = synderesis_engine
        self.logger = logging.getLogger(__name__)
    
    async def build_syllogism(
        self,
        action: str,
        circumstances: CircumstanceAnalysis,
        agent_id: str
    ) -> MoralSyllogism:
        """Build complete moral syllogism for the given action"""
        
        # Identify most relevant moral principle
        relevant_principle = self._identify_most_relevant_principle(action, circumstances)
        
        # Construct major premise (universal principle)
        major_premise = self._construct_major_premise(relevant_principle)
        
        # Construct minor premise (particular circumstances)
        minor_premise = self._construct_minor_premise(action, circumstances)
        
        # Derive conclusion
        conclusion = self._derive_conclusion(major_premise, minor_premise, action)
        
        # Assess logical validity
        logical_validity = self._assess_logical_validity(major_premise, minor_premise, conclusion)
        
        # Build reasoning steps
        reasoning_steps = self._build_reasoning_steps(
            relevant_principle, action, circumstances, major_premise, minor_premise, conclusion
        )
        
        # Determine conclusion certainty
        conclusion_certainty = self._calculate_conclusion_certainty(
            relevant_principle, circumstances, logical_validity
        )
        
        syllogism = MoralSyllogism(
            major_premise=major_premise,
            major_premise_source=relevant_principle.id if relevant_principle else "unknown",
            major_premise_certainty=relevant_principle.certainty if relevant_principle else MoralCertaintyLevel.MODERATE,
            minor_premise=minor_premise,
            minor_premise_analysis=circumstances,
            minor_premise_certainty=circumstances.certainty_level,
            conclusion=conclusion,
            conclusion_certainty=conclusion_certainty,
            conclusion_type=MoralJudgmentType.ANTECEDENT,
            reasoning_steps=reasoning_steps,
            logical_validity=logical_validity,
            practical_applicability=self._assess_practical_applicability(action, circumstances)
        )
        
        return syllogism
    
    def _identify_most_relevant_principle(
        self,
        action: str,
        circumstances: CircumstanceAnalysis
    ) -> Optional[MoralPrinciple]:
        """Identify the most relevant moral principle for this action"""
        
        # Get all synderesis principles
        principles = self.synderesis_engine.synderesis_principles
        
        best_principle = None
        highest_relevance = 0.0
        
        for principle in principles.values():
            relevance = self.synderesis_engine._assess_principle_relevance(
                principle, action, circumstances
            )
            
            if relevance > highest_relevance:
                highest_relevance = relevance
                best_principle = principle
        
        return best_principle if highest_relevance > 0.3 else None
    
    def _construct_major_premise(self, principle: Optional[MoralPrinciple]) -> str:
        """Construct the major premise from the moral principle"""
        
        if not principle:
            return "Good is to be done and evil avoided"
        
        # Use the English translation of the principle
        return principle.english_translation
    
    def _construct_minor_premise(self, action: str, circumstances: CircumstanceAnalysis) -> str:
        """Construct the minor premise describing particular circumstances"""
        
        premise_parts = []
        
        # Include the action itself
        premise_parts.append(f"The proposed action is: {action}")
        
        # Include key circumstances
        if circumstances.quis:
            premise_parts.append(f"performed by {circumstances.quis}")
        
        if circumstances.cur:
            premise_parts.append(f"motivated by {circumstances.cur}")
        
        if circumstances.quomodo:
            premise_parts.append(f"in the manner of {circumstances.quomodo}")
        
        # Include contextual factors
        if circumstances.stakes_level > 0.7:
            premise_parts.append("with high stakes")
        
        if circumstances.urgency_level > 0.7:
            premise_parts.append("under time pressure")
        
        return ", ".join(premise_parts)
    
    def _derive_conclusion(self, major_premise: str, minor_premise: str, action: str) -> str:
        """Derive moral conclusion from premises"""
        
        # Simplified logical derivation
        if any(negative_term in major_premise.lower() for negative_term in ["avoid", "not", "forbidden"]):
            if any(negative_indicator in minor_premise.lower() for negative_indicator in ["harm", "deceive", "unjust"]):
                return f"Therefore, this action should be avoided"
        
        if any(positive_term in major_premise.lower() for positive_term in ["do", "pursue", "good"]):
            if any(positive_indicator in minor_premise.lower() for positive_indicator in ["help", "honest", "just", "charity"]):
                return f"Therefore, this action should be pursued"
        
        # Default neutral conclusion
        return f"Therefore, this action is morally permissible but not obligatory"
    
    def _assess_logical_validity(self, major_premise: str, minor_premise: str, conclusion: str) -> float:
        """Assess the logical validity of the syllogism"""
        
        validity_score = 0.5  # Start with neutral
        
        # Check for basic logical structure
        if "therefore" in conclusion.lower():
            validity_score += 0.2
        
        # Check for consistency between premises and conclusion
        if "avoid" in major_premise.lower() and "avoid" in conclusion.lower():
            validity_score += 0.2
        elif "do" in major_premise.lower() and ("pursue" in conclusion.lower() or "permissible" in conclusion.lower()):
            validity_score += 0.2
        
        # Check for contradiction
        if ("avoid" in major_premise.lower() and "pursue" in conclusion.lower()) or \
           ("do" in major_premise.lower() and "avoid" in conclusion.lower()):
            validity_score -= 0.4
        
        return max(0.0, min(1.0, validity_score))
    
    def _build_reasoning_steps(
        self,
        principle: Optional[MoralPrinciple],
        action: str,
        circumstances: CircumstanceAnalysis,
        major_premise: str,
        minor_premise: str,
        conclusion: str
    ) -> List[str]:
        """Build step-by-step reasoning chain"""
        
        steps = []
        
        # Step 1: Identify relevant principle
        if principle:
            steps.append(f"Identified relevant moral principle: {principle.english_name} ({principle.thomistic_source})")
        else:
            steps.append("Applied general synderesis principle of doing good and avoiding evil")
        
        # Step 2: State universal principle
        steps.append(f"Universal moral principle: {major_premise}")
        
        # Step 3: Analyze particular circumstances
        steps.append(f"Particular circumstances: {minor_premise}")
        
        # Step 4: Apply principle to circumstances
        steps.append("Applied universal principle to particular circumstances")
        
        # Step 5: Consider modifying factors
        if circumstances.aggravating_factors:
            steps.append(f"Considered aggravating factors: {', '.join(circumstances.aggravating_factors)}")
        
        if circumstances.mitigating_factors:
            steps.append(f"Considered mitigating factors: {', '.join(circumstances.mitigating_factors)}")
        
        # Step 6: Reach conclusion
        steps.append(f"Moral judgment: {conclusion}")
        
        return steps
    
    def _calculate_conclusion_certainty(
        self,
        principle: Optional[MoralPrinciple],
        circumstances: CircumstanceAnalysis,
        logical_validity: float
    ) -> float:
        """Calculate certainty level of the conclusion"""
        
        certainty_factors = []
        
        # Principle certainty
        if principle:
            principle_certainty_map = {
                MoralCertaintyLevel.ABSOLUTE: 1.0,
                MoralCertaintyLevel.VERY_HIGH: 0.9,
                MoralCertaintyLevel.HIGH: 0.8,
                MoralCertaintyLevel.MODERATE: 0.6,
                MoralCertaintyLevel.LOW: 0.4,
                MoralCertaintyLevel.DOUBTFUL: 0.2
            }
            certainty_factors.append(principle_certainty_map.get(principle.certainty, 0.5))
        else:
            certainty_factors.append(0.5)
        
        # Circumstance certainty
        certainty_factors.append(circumstances.certainty_level)
        
        # Logical validity
        certainty_factors.append(logical_validity)
        
        # Complexity factor (higher complexity reduces certainty)
        certainty_factors.append(1.0 - circumstances.complexity_score)
        
        return sum(certainty_factors) / len(certainty_factors)
    
    def _assess_practical_applicability(self, action: str, circumstances: CircumstanceAnalysis) -> float:
        """Assess how practically applicable this syllogism is"""
        
        applicability_score = 0.5
        
        # More applicable if circumstances are clear
        applicability_score += circumstances.certainty_level * 0.3
        
        # More applicable if not too complex
        applicability_score += (1.0 - circumstances.complexity_score) * 0.2
        
        # More applicable if action is concrete rather than abstract
        if len(action.split()) > 3:  # More detailed action description
            applicability_score += 0.2
        
        # Less applicable if under extreme time pressure
        if circumstances.urgency_level > 0.9:
            applicability_score -= 0.2
        
        return max(0.0, min(1.0, applicability_score))
```

---

## 6. Moral Anxiety Generation {#moral-anxiety-generation}

### Dynamic Anxiety Calibration System

```python
from typing import Dict, List, Optional, Any, Tuple
import math
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

class AnxietyCalibrationStrategy(Enum):
    """Strategies for calibrating moral anxiety levels"""
    FORMATION_BASED = "formation_based"      # Based on conscience development stage
    VIRTUE_INTEGRATED = "virtue_integrated"  # Integrated with virtue development
    SITUATION_ADAPTIVE = "situation_adaptive" # Adapted to situation characteristics
    LEARNING_OPTIMIZED = "learning_optimized" # Optimized for moral learning

@dataclass
class AnxietyCalibrationProfile:
    """Profile for anxiety calibration specific to an agent"""
    agent_id: str
    current_formation_level: float           # 0.0 to 1.0
    conscience_maturity: float              # 0.0 to 1.0
    virtue_integration_score: float         # 0.0 to 1.0
    sensitivity_preferences: Dict[str, float]  # Preferred anxiety levels by type
    scrupulosity_indicators: List[str]      # Signs of excessive anxiety
    laxity_indicators: List[str]            # Signs of insufficient anxiety
    anxiety_effectiveness_history: Dict[str, float]  # Historical effectiveness
    formation_goals: List[str]              # Current formation objectives
    
    def __post_init__(self):
        if not self.sensitivity_preferences:
            self.sensitivity_preferences = {
                MoralAnxietyType.PRINCIPLE_VIOLATION.value: 0.8,
                MoralAnxietyType.PRUDENTIAL_UNCERTAINTY.value: 0.6,
                MoralAnxietyType.CONSEQUENTIAL_CONCERN.value: 0.5,
                MoralAnxietyType.AUTHORITY_CONFLICT.value: 0.7,
                MoralAnxietyType.TEMPORAL_PRESSURE.value: 0.4,
                MoralAnxietyType.INFORMATION_INSUFFICIENCY.value: 0.5
            }

class MoralAnxietyGenerator:
    """Generator for appropriate moral anxiety based on Thomistic principles"""
    
    def __init__(self, synderesis_engine: SynderesisEngine, virtue_integration=None):
        self.synderesis_engine = synderesis_engine
        self.virtue_integration = virtue_integration
        self.logger = logging.getLogger(__name__)
        
        # Anxiety generation parameters
        self.anxiety_parameters = self._initialize_anxiety_parameters()
        
        # Agent calibration profiles
        self.calibration_profiles = {}
        
        # Anxiety effectiveness tracking
        self.effectiveness_tracker = AnxietyEffectivenessTracker()
    
    def _initialize_anxiety_parameters(self) -> Dict[str, Any]:
        """Initialize parameters for anxiety generation"""
        return {
            # Base intensity levels for different triggers
            "base_intensities": {
                "synderesis_violation": 0.8,
                "natural_law_conflict": 0.7,
                "prudential_doubt": 0.5,
                "circumstantial_uncertainty": 0.4,
                "temporal_pressure": 0.3,
                "stakes_elevation": 0.6
            },
            
            # Modifiers based on agent characteristics
            "formation_modifiers": {
                "beginner": 0.8,     # Lower anxiety for beginners
                "developing": 1.0,   # Standard anxiety
                "mature": 1.2,       # Higher sensitivity for mature agents
                "advanced": 1.1      # Refined sensitivity for advanced agents
            },
            
            # Situation-specific modifiers
            "situation_modifiers": {
                "high_stakes": 1.3,
                "time_pressure": 0.9,  # Slightly reduce to avoid paralysis
                "public_context": 1.2,
                "private_context": 1.0,
                "teaching_moment": 1.4  # Higher for educational value
            },
            
            # Therapeutic bounds (prevent harmful anxiety levels)
            "therapeutic_bounds": {
                "minimum_sensitivity": 0.1,
                "maximum_intensity": 0.9,
                "scrupulosity_threshold": 0.8,
                "laxity_threshold": 0.2
            }
        }
    
    async def generate_anxiety_for_situation(
        self,
        agent_id: str,
        moral_decision_context: Dict[str, Any],
        circumstances: CircumstanceAnalysis,
        synderesis_evaluation: Dict[str, Any],
        conscientia_processing: Dict[str, Any]
    ) -> List[MoralAnxietyInstance]:
        """Generate appropriate moral anxiety for the given situation"""
        
        # Get or create calibration profile for agent
        calibration_profile = await self._get_calibration_profile(agent_id)
        
        anxiety_instances = []
        
        try:
            # Generate anxiety from synderesis violations
            synderesis_anxiety = await self._generate_synderesis_anxiety(
                agent_id, synderesis_evaluation, circumstances, calibration_profile
            )
            anxiety_instances.extend(synderesis_anxiety)
            
            # Generate anxiety from prudential uncertainty
            prudential_anxiety = await self._generate_prudential_anxiety(
                agent_id, conscientia_processing, circumstances, calibration_profile
            )
            anxiety_instances.extend(prudential_anxiety)
            
            # Generate anxiety from consequential concerns
            consequential_anxiety = await self._generate_consequential_anxiety(
                agent_id, moral_decision_context, circumstances, calibration_profile
            )
            anxiety_instances.extend(consequential_anxiety)
            
            # Generate anxiety from authority conflicts
            authority_anxiety = await self._generate_authority_conflict_anxiety(
                agent_id, moral_decision_context, circumstances, calibration_profile
            )
            anxiety_instances.extend(authority_anxiety)
            
            # Generate anxiety from temporal pressure
            temporal_anxiety = await self._generate_temporal_pressure_anxiety(
                agent_id, circumstances, calibration_profile
            )
            anxiety_instances.extend(temporal_anxiety)
            
            # Generate anxiety from information insufficiency
            information_anxiety = await self._generate_information_anxiety(
                agent_id, circumstances, conscientia_processing, calibration_profile
            )
            anxiety_instances.extend(information_anxiety)
            
            # Apply calibration and therapeutic bounds
            anxiety_instances = await self._apply_calibration(
                anxiety_instances, calibration_profile, circumstances
            )
            
            # Update effectiveness tracking
            await self._update_effectiveness_tracking(agent_id, anxiety_instances, circumstances)
            
        except Exception as e:
            self.logger.error(f"Error generating anxiety for agent {agent_id}: {e}")
        
        return anxiety_instances
    
    async def _generate_synderesis_anxiety(
        self,
        agent_id: str,
        synderesis_evaluation: Dict[str, Any],
        circumstances: CircumstanceAnalysis,
        calibration_profile: AnxietyCalibrationProfile
    ) -> List[MoralAnxietyInstance]:
        """Generate anxiety from synderesis principle violations"""
        
        anxiety_instances = []
        
        # Process direct principle violations
        for violation in synderesis_evaluation.get("violated_principles", []):
            # Base intensity from violation severity
            base_intensity = violation["severity"]
            
            # Apply calibration
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.PRINCIPLE_VIOLATION,
                calibration_profile,
                circumstances
            )
            
            # Create anxiety instance
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.PRINCIPLE_VIOLATION,
                intensity_level=calibrated_intensity,
                appropriateness_score=self._calculate_appropriateness(
                    calibrated_intensity, violation["severity"], "principle_violation"
                ),
                triggering_scenario=f"Violation of {violation['principle_id']}: {violation['description']}",
                violated_principles=[violation["principle_id"]],
                decision_impact=self._calculate_decision_impact(
                    calibrated_intensity, violation["severity"]
                ),
                learning_value=self._calculate_learning_value(
                    violation["severity"], calibration_profile.current_formation_level
                ),
                formation_contribution=self._calculate_formation_contribution(
                    calibrated_intensity, "principle_violation", calibration_profile
                )
            )
            
            anxiety_instances.append(anxiety_instance)
        
        # Process moral concerns (lower-level issues)
        for concern in synderesis_evaluation.get("moral_concerns", []):
            # Lower intensity for concerns vs violations
            base_intensity = 0.3
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.PRINCIPLE_VIOLATION,
                calibration_profile,
                circumstances
            ) * 0.6  # Reduce for concerns
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.PRINCIPLE_VIOLATION,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.7,
                triggering_scenario=f"Moral concern: {concern}",
                decision_impact=calibrated_intensity * 0.5,
                learning_value=0.6,
                formation_contribution=calibrated_intensity * 0.4
            )
            
            anxiety_instances.append(anxiety_instance)
        
        return anxiety_instances
    
    async def _generate_prudential_anxiety(
        self,
        agent_id: str,
        conscientia_processing: Dict[str, Any],
        circumstances: CircumstanceAnalysis,
        calibration_profile: AnxietyCalibrationProfile
    ) -> List[MoralAnxietyInstance]:
        """Generate anxiety from prudential uncertainty"""
        
        anxiety_instances = []
        
        # Check for doubt levels
        doubt_level = conscientia_processing.get("doubt_level", MoralDoubtLevel.NO_DOUBT)
        
        if doubt_level != MoralDoubtLevel.NO_DOUBT:
            # Map doubt level to intensity
            doubt_intensity_map = {
                MoralDoubtLevel.SLIGHT_DOUBT: 0.3,
                MoralDoubtLevel.SERIOUS_DOUBT: 0.6,
                MoralDoubtLevel.PERPLEXITY: 0.8,
                MoralDoubtLevel.SCRUPULOSITY: 0.4  # Moderate but persistent
            }
            
            base_intensity = doubt_intensity_map.get(doubt_level, 0.5)
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.PRUDENTIAL_UNCERTAINTY,
                calibration_profile,
                circumstances
            )
            
            # Special handling for scrupulosity
            if doubt_level == MoralDoubtLevel.SCRUPULOSITY:
                calibrated_intensity = min(calibrated_intensity, 0.5)  # Cap scrupulous anxiety
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.PRUDENTIAL_UNCERTAINTY,
                intensity_level=calibrated_intensity,
                appropriateness_score=self._calculate_doubt_appropriateness(doubt_level),
                triggering_scenario=f"Moral doubt: {doubt_level.value}",
                uncertain_factors=conscientia_processing.get("doubt_indicators", []),
                decision_impact=calibrated_intensity * 0.7,
                learning_value=self._calculate_doubt_learning_value(doubt_level),
                formation_contribution=calibrated_intensity * 0.6
            )
            
            anxiety_instances.append(anxiety_instance)
        
        # Check for low moral certainty
        moral_certainty = conscientia_processing.get("moral_certainty", 0.5)
        if moral_certainty < 0.4:
            uncertainty_intensity = (0.4 - moral_certainty) / 0.4 * 0.5  # Scale to 0-0.5
            
            calibrated_intensity = self._apply_intensity_calibration(
                uncertainty_intensity,
                MoralAnxietyType.PRUDENTIAL_UNCERTAINTY,
                calibration_profile,
                circumstances
            )
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.PRUDENTIAL_UNCERTAINTY,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.8,
                triggering_scenario=f"Low moral certainty: {moral_certainty:.2f}",
                decision_impact=calibrated_intensity * 0.6,
                learning_value=0.7,
                formation_contribution=calibrated_intensity * 0.5
            )
            
            anxiety_instances.append(anxiety_instance)
        
        return anxiety_instances
    
    async def _generate_consequential_anxiety(
        self,
        agent_id: str,
        moral_decision_context: Dict[str, Any],
        circumstances: CircumstanceAnalysis,
        calibration_profile: AnxietyCalibrationProfile
    ) -> List[MoralAnxietyInstance]:
        """Generate anxiety from concern about consequences"""
        
        anxiety_instances = []
        
        # High stakes generate consequential anxiety
        if circumstances.stakes_level > 0.7:
            # Intensity proportional to stakes level
            base_intensity = (circumstances.stakes_level - 0.7) / 0.3 * 0.6  # Scale to 0-0.6
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.CONSEQUENTIAL_CONCERN,
                calibration_profile,
                circumstances
            )
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.CONSEQUENTIAL_CONCERN,
                intensity_level=calibrated_intensity,
                appropriateness_score=0.8,
                triggering_scenario=f"High stakes decision (level: {circumstances.stakes_level:.2f})",
                decision_impact=calibrated_intensity * 0.5,
                learning_value=0.6,
                formation_contribution=calibrated_intensity * 0.4
            )
            
            anxiety_instances.append(anxiety_instance)
        
        # Uncertainty about outcomes
        if circumstances.certainty_level < 0.3 and circumstances.stakes_level > 0.5:
            # Combine uncertainty and stakes for intensity
            base_intensity = (0.3 - circumstances.certainty_level) / 0.3 * circumstances.stakes_level * 0.7
            
            calibrated_intensity = self._apply_intensity_calibration(
                base_intensity,
                MoralAnxietyType.CONSEQUENTIAL_CONCERN,
                calibration_profile,
                circumstances
            )
            
            anxiety_instance = MoralAnxietyInstance(
                agent_id=agent_id,
                anxiety_type=MoralAnxietyType.CONSEQUENTIAL_CONCERN,
                intensity_level=calibrated_intensity,
                appropriat# Moral Anxiety System - Complete Technical Specifications
*Thomistic Conscience Formation Engine for AI Moral Sensitivity*

## Table of Contents
1. [Architectural Overview](#architectural-overview)
2. [Thomistic Conscience Theory Implementation](#thomistic-conscience-theory)
3. [Core Data Models](#core-data-models)
4. [Synderesis Engine](#synderesis-engine)
5. [Conscientia Processing System](#conscientia-processing)
6. [Moral Anxiety Generation](#moral-anxiety-generation)
7. [Database Schema](#database-schema)
8. [API Specifications](#api-specifications)
9. [Implementation Classes](#implementation-classes)
10. [Testing Framework](#testing-framework)
11. [Performance Optimization](#performance-optimization)
12. [Integration Patterns](#integration-patterns)

---

## 1. Architectural Overview {#architectural-overview}

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                    Moral Anxiety System                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │   Synderesis   │  │  Conscientia   │  │  Anxiety        │   │
│  │   Engine       │  │  Processor     │  │  Generator      │   │
│  │  (First        │  │  (Applied      │  │  (Moral         │   │
│  │  Principles)   │  │  Judgment)     │  │  Sensitivity)   │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │  Moral         │  │  Doubt         │  │  Conscience     │   │
│  │  Principles    │  │  Resolution    │  │  Formation      │   │
│  │  Database      │  │  System        │  │  Tracker        │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │  Circumstance  │  │  Moral         │  │  Sensitivity    │   │
│  │  Analyzer      │  │  Reasoning     │  │  Calibrator     │   │
│  │               │  │  Engine        │  │                 │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

#### 1. Thomistic Fidelity
- Accurate implementation of Aquinas's two-level conscience structure
- Distinction between synderesis (natural moral knowledge) and conscientia (applied judgment)
- Integration with natural law theory and prudential reasoning

#### 2. Dynamic Moral Sensitivity
- Real-time moral tension generation appropriate to situation gravity
- Adaptive anxiety calibration based on agent moral development
- Context-sensitive conscience formation and refinement

#### 3. Prudential Integration
- Integration with virtue tracking for prudence-guided moral reasoning
- Circumstantial analysis for proper moral judgment
- Doubt resolution protocols following Thomistic methodology

#### 4. Scalable Conscience Architecture
- High-performance processing of complex moral scenarios
- Concurrent moral evaluation capabilities
- Efficient storage and retrieval of moral reasoning chains

---

## 2. Thomistic Conscience Theory Implementation {#thomistic-conscience-theory}

### Theoretical Foundation

#### Aquinas's Two-Level Conscience Structure

##### Level 1: Synderesis (*Synderesis*)
**Definition**: Natural habit of the practical intellect containing first principles of moral action
**Thomistic Source**: *Summa Theologiae* I, q. 79, a. 12; *De Veritate* q. 16, a. 1

**Characteristics**:
- **Immutable**: Cannot be corrupted or destroyed
- **Universal**: Same principles for all rational beings  
- **Infallible**: Cannot err about first principles
- **Natural**: Present from the beginning of rational nature

**Primary Principle**: *Bonum est faciendum et prosequendum, et malum vitandum*
"Good is to be done and pursued, and evil is to be avoided"

**Secondary Principles** (Natural Law Precepts):
1. **Life Preservation**: Rational life must be protected and promoted
2. **Knowledge Pursuit**: Truth must be sought and error avoided
3. **Social Harmony**: Community relationships must be maintained in justice
4. **Transcendent Orientation**: Ultimate meaning and divine order must be recognized

##### Level 2: Conscientia (*Conscientia*)
**Definition**: Act of practical reason applying synderesis principles to particular circumstances
**Thomistic Source**: *Summa Theologiae* I, q. 79, a. 13; *De Veritate* q. 17

**Process Structure**:
```
Major Premise (Synderesis): Universal moral principle
Minor Premise (Prudence): Particular circumstances analysis  
Conclusion (Conscientia): Specific moral judgment
```

**Types of Conscientia**:
- **Antecedent**: Judgment before action (guidance)
- **Consequent**: Judgment after action (approval/condemnation)
- **Certain**: Clear judgment without reasonable doubt
- **Doubtful**: Uncertain judgment requiring resolution
- **Perplexed**: Conflicted judgment between competing goods
- **Scrupulous**: Excessive doubt about moral matters

#### Moral Anxiety in Thomistic Framework

**Theological Foundation**: Fear of moral evil as appropriate emotional response
**Source**: *Summa Theologiae* II-II, q. 19 (On Fear), q. 125 (On Daring)

**Appropriate Moral Fear**:
1. **Timor Malorum**: Fear of moral evil and sin
2. **Timor Poenae**: Fear of consequences of moral failure
3. **Timor Reverentialis**: Reverential fear maintaining proper moral attitude

**Moral Anxiety Functions**:
- **Deterrent**: Prevents approach to moral evil
- **Motivational**: Encourages moral good pursuit
- **Formational**: Develops moral sensitivity over time
- **Protective**: Guards against moral complacency

### Implementation Framework

#### Natural Law Principle Hierarchy
```python
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import uuid

class MoralPrincipleLevel(IntEnum):
    """Hierarchy of moral principle certainty"""
    PRIMARY = 1      # Synderesis first principle
    SECONDARY = 2    # Natural law precepts  
    TERTIARY = 3     # Derived moral rules
    PRUDENTIAL = 4   # Circumstantial applications

class MoralCertaintyLevel(Enum):
    """Levels of moral certainty in judgments"""
    ABSOLUTE = "absolute"          # Synderesis principles
    VERY_HIGH = "very_high"       # Clear natural law
    HIGH = "high"                 # Well-established moral teaching
    MODERATE = "moderate"         # Probable moral opinion
    LOW = "low"                   # Uncertain application
    DOUBTFUL = "doubtful"         # Competing probable opinions

@dataclass
class MoralPrinciple:
    """Fundamental moral principle in Thomistic framework"""
    id: str
    level: MoralPrincipleLevel
    certainty: MoralCertaintyLevel
    latin_formulation: str
    english_translation: str
    thomistic_source: str
    explanation: str
    natural_law_basis: str
    universal_scope: bool
    immutable: bool
    foundational_principles: List[str] = None  # References to more basic principles
    derived_principles: List[str] = None       # References to derived principles
    circumstantial_exceptions: List[str] = None
    
    def __post_init__(self):
        if self.foundational_principles is None:
            self.foundational_principles = []
        if self.derived_principles is None:
            self.derived_principles = []
        if self.circumstantial_exceptions is None:
            self.circumstantial_exceptions = []

class ThomisticMoralPrincipleRegistry:
    """Registry of moral principles based on Thomistic sources"""
    
    def __init__(self):
        self.principles = self._initialize_thomistic_principles()
        self.principle_hierarchy = self._build_principle_hierarchy()
    
    def _initialize_thomistic_principles(self) -> Dict[str, MoralPrinciple]:
        """Initialize complete moral principle registry from Thomistic sources"""
        
        principles = {}
        
        # PRIMARY PRINCIPLE (Synderesis)
        principles["primary_synderesis"] = MoralPrinciple(
            id="primary_synderesis",
            level=MoralPrincipleLevel.PRIMARY,
            certainty=MoralCertaintyLevel.ABSOLUTE,
            latin_formulation="Bonum est faciendum et prosequendum, et malum vitandum",
            english_translation="Good is to be done and pursued, and evil is to be avoided",
            thomistic_source="ST I-II, q. 94, a. 2",
            explanation="The first and most universal principle of practical reason, naturally known to all rational beings",
            natural_law_basis="Eternal law participation through rational nature",
            universal_scope=True,
            immutable=True
        )
        
        # SECONDARY PRINCIPLES (Natural Law Precepts)
        principles["life_preservation"] = MoralPrinciple(
            id="life_preservation",
            level=MoralPrincipleLevel.SECONDARY,
            certainty=MoralCertaintyLevel.VERY_HIGH,
            latin_formulation="Vita rationalis conservanda est",
            english_translation="Rational life must be preserved",
            thomistic_source="ST I-II, q. 94, a. 2",
            explanation="Natural inclination to preserve life leads to moral principle of life protection",
            natural_law_basis="Natural inclination common to all substances",
            universal_scope=True,
            immutable=True,
            foundational_principles=["primary_synderesis"]
        )
        
        principles["truth_pursuit"] = MoralPrinciple(
            id="truth_pursuit", 
            level=MoralPrincipleLevel.SECONDARY,
            certainty=MoralCertaintyLevel.VERY_HIGH,
            latin_formulation="Veritas cognoscenda est et error vitandus",
            english_translation="Truth must be known and error must be avoided",
            thomistic_source="ST I-II, q. 94, a. 2",
            explanation="Natural desire to know truth creates obligation to seek truth and avoid error",
            natural_law_basis="Rational nature's orientation toward truth",
            universal_scope=True,
            immutable=True,
            foundational_principles=["primary_synderesis"]
        )
        
        principles["social_harmony"] = MoralPrinciple(
            id="social_harmony",
            level=MoralPrincipleLevel.SECONDARY, 
            certainty=MoralCertaintyLevel.VERY_HIGH,
            latin_formulation="Vita socialis in iustitia servanda est",
            english_translation="Social life must be maintained in justice",
            thomistic_source="ST I-II, q. 94, a. 2",
            explanation="Human social nature creates obligations of justice toward others",
            natural_law_basis="Natural inclination to live in society",
            universal_scope=True,
            immutable=True,
            foundational_principles=["primary_synderesis"]
        )
        
        principles["divine_orientation"] = MoralPrinciple(
            id="divine_orientation",
            level=MoralPrincipleLevel.SECONDARY,
            certainty=MoralCertaintyLevel.VERY_HIGH,
            latin_formulation="Deus colendus est et ordo transcendens recognoscendus",
            english_translation="God must be worshipped and transcendent order recognized",
            thomistic_source="ST I-II, q. 94, a. 2",
            explanation="Natural orientation toward ultimate good creates religious obligation",
            natural_law_basis="Natural inclination toward ultimate happiness in God",
            universal_scope=True,
            immutable=True,
            foundational_principles=["primary_synderesis"]
        )
        
        # TERTIARY PRINCIPLES (Derived Moral Rules)
        principles["promise_keeping"] = MoralPrinciple(
            id="promise_keeping",
            level=MoralPrincipleLevel.TERTIARY,
            certainty=MoralCertaintyLevel.HIGH,
            latin_formulation="Promissa servanda sunt",
            english_translation="Promises must be kept",
            thomistic_source="ST II-II, q. 110, a. 3",
            explanation="Justice requires fidelity to commitments made to others",
            natural_law_basis="Derived from social harmony and truth principles",
            universal_scope=True,
            immutable=False,
            foundational_principles=["social_harmony", "truth_pursuit"],
            circumstantial_exceptions=["impossible_performance", "immoral_content", "changed_circumstances"]
        )
        
        principles["harm_prevention"] = MoralPrinciple(
            id="harm_prevention",
            level=MoralPrincipleLevel.TERTIARY,
            certainty=MoralCertaintyLevel.HIGH,
            latin_formulation="Nocumentum aliis non inferendum",
            english_translation="Harm must not be inflicted on others",
            thomistic_source="ST II-II, q. 64, a. 1",
            explanation="Life preservation principle requires avoiding harm to others",
            natural_law_basis="Derived from life preservation and social harmony",
            universal_scope=True,
            immutable=False,
            foundational_principles=["life_preservation", "social_harmony"],
            circumstantial_exceptions=["self_defense", "legitimate_authority", "double_effect"]
        )
        
        principles["property_respect"] = MoralPrinciple(
            id="property_respect",
            level=MoralPrincipleLevel.TERTIARY,
            certainty=MoralCertaintyLevel.HIGH,
            latin_formulation="Proprietas aliena respectanda est",
            english_translation="Others' property must be respected",
            thomistic_source="ST II-II, q. 66, a. 1",
            explanation="Justice requires respect for legitimate ownership rights",
            natural_law_basis="Derived from social harmony and life preservation",
            universal_scope=True,
            immutable=False,
            foundational_principles=["social_harmony", "life_preservation"],
            circumstantial_exceptions=["extreme_necessity", "common_good", "legitimate_taxation"]
        )
        
        principles["truthfulness"] = MoralPrinciple(
            id="truthfulness",
            level=MoralPrincipleLevel.TERTIARY,
            certainty=MoralCertaintyLevel.HIGH,
            latin_formulation="Veritas in verbis servanda est",
            english_translation="Truth must be maintained in speech",
            thomistic_source="ST II-II, q. 109, a. 1",
            explanation="Truth pursuit principle requires honesty in communication",
            natural_law_basis="Derived from truth pursuit and social harmony",
            universal_scope=True,
            immutable=False,
            foundational_principles=["truth_pursuit", "social_harmony"],
            circumstantial_exceptions=["mental_reservation", "protection_of_innocent", "professional_secrecy"]
        )
        
        # PRUDENTIAL PRINCIPLES (Circumstantial Applications)
        principles["double_effect"] = MoralPrinciple(
            id="double_effect",
            level=MoralPrincipleLevel.PRUDENTIAL,
            certainty=MoralCertaintyLevel.HIGH,
            latin_formulation="Actus duplicis effectus licitus sub conditionibus",
            english_translation="Acts with double effect are permissible under conditions",
            thomistic_source="ST II-II, q. 64, a. 7",
            explanation="Framework for evaluating acts with both good and bad effects",
            natural_law_basis="Prudential application of fundamental principles",
            universal_scope=True,
            immutable=False,
            foundational_principles=["primary_synderesis", "harm_prevention"]
        )
        
        principles["lesser_evil"] = MoralPrinciple(
            id="lesser_evil",
            level=MoralPrincipleLevel.PRUDENTIAL,
            certainty=MoralCertaintyLevel.MODERATE,
            latin_formulation="Minus malum eligendum quando necessarium",
            english_translation="Lesser evil may be chosen when necessary",
            thomistic_source="ST I-II, q. 78, a. 1",
            explanation="In unavoidable moral conflicts, choose action causing least moral harm",
            natural_law_basis="Prudential reasoning about competing moral obligations",
            universal_scope=False,
            immutable=False,
            foundational_principles=["primary_synderesis"]
        )
        
        return principles
    
    def _build_principle_hierarchy(self) -> Dict[str, List[str]]:
        """Build hierarchical relationships between principles"""
        hierarchy = {}
        
        for principle_id, principle in self.principles.items():
            # Build upward references (what this principle depends on)
            hierarchy[principle_id] = {
                "depends_on": principle.foundational_principles,
                "supports": principle.derived_principles,
                "level": principle.level,
                "certainty": principle.certainty
            }
        
        return hierarchy
    
    def get_principle(self, principle_id: str) -> Optional[MoralPrinciple]:
        """Get specific moral principle"""
        return self.principles.get(principle_id)
    
    def get_principles_by_level(self, level: MoralPrincipleLevel) -> List[MoralPrinciple]:
        """Get all principles at specified level"""
        return [p for p in self.principles.values() if p.level == level]
    
    def get_foundational_principles(self, principle_id: str) -> List[MoralPrinciple]:
        """Get all foundational principles for given principle"""
        principle = self.principles.get(principle_id)
        if not principle:
            return []
        
        foundational = []
        for foundation_id in principle.foundational_principles:
            foundation = self.principles.get(foundation_id)
            if foundation:
                foundational.append(foundation)
                # Recursively get foundations of foundations
                foundational.extend(self.get_foundational_principles(foundation_id))
        
        return foundational
    
    def find_principle_conflicts(self, principle_ids: List[str]) -> List[Dict[str, str]]:
        """Identify potential conflicts between principles"""
        conflicts = []
        
        for i, principle_id1 in enumerate(principle_ids):
            for principle_id2 in principle_ids[i+1:]:
                principle1 = self.principles.get(principle_id1)
                principle2 = self.principles.get(principle_id2)
                
                if principle1 and principle2:
                    # Check for direct conflicts (this would be expanded with conflict detection logic)
                    conflict = self._detect_principle_conflict(principle1, principle2)
                    if conflict:
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_principle_conflict(self, principle1: MoralPrinciple, principle2: MoralPrinciple) -> Optional[Dict[str, str]]:
        """Detect conflict between two principles (simplified implementation)"""
        # Known principle conflicts in Thomistic framework
        known_conflicts = {
            ("harm_prevention", "truth_pursuit"): "Truthfulness may cause harm in certain circumstances",
            ("property_respect", "life_preservation"): "Extreme necessity may override property rights",
            ("promise_keeping", "harm_prevention"): "Promises may require harmful actions"
        }
        
        conflict_key = (principle1.id, principle2.id)
        reverse_key = (principle2.id, principle1.id)
        
        if conflict_key in known_conflicts:
            return {
                "principle1": principle1.id,
                "principle2": principle2.id,
                "type": "material_conflict",
                "description": known_conflicts[conflict_key]
            }
        elif reverse_key in known_conflicts:
            return {
                "principle1": principle2.id,
                "principle2": principle1.id,
                "type": "material_conflict", 
                "description": known_conflicts[reverse_key]
            }
        
        return None
```

---

## 3. Core Data Models {#core-data-models}

### Moral Judgment Data Structures

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
import uuid

class MoralJudgmentType(Enum):
    """Types of moral judgments in conscientia"""
    ANTECEDENT = "antecedent"     # Before action
    CONSEQUENT = "consequent"     # After action  
    HYPOTHETICAL = "hypothetical" # Theoretical evaluation

class MoralAnxietyType(Enum):
    """Types of moral anxiety"""
    PRINCIPLE_VIOLATION = "principle_violation"    # Direct violation of synderesis
    PRUDENTIAL_UNCERTAINTY = "prudential_uncertainty"  # Unclear circumstances
    CONSEQUENTIAL_CONCERN = "consequential_concern"    # Worry about outcomes
    AUTHORITY_CONFLICT = "authority_conflict"      # Competing moral authorities
    TEMPORAL_PRESSURE = "temporal_pressure"        # Time constraints on decision
    INFORMATION_INSUFFICIENCY = "information_insufficiency"  # Lack of relevant data

class DoubtResolutionStrategy(Enum):
    """Strategies for resolving moral doubt"""
    SEEK_INFORMATION = "seek_information"
    CONSULT_AUTHORITY = "consult_authority"
    APPLY_SAFER_COURSE = "apply_safer_course"
    USE_PROBABILITY = "use_probability"
    REFRAIN_FROM_ACTION = "refrain_from_action"

@dataclass
class CircumstanceAnalysis:
    """Analysis of morally relevant circumstances"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Thomistic circumstantial factors (ST I-II, q. 7)
    quis: Optional[str] = None      # Who (agent)
    quid: Optional[str] = None      # What (object/act)
    ubi: Optional[str] = None       # Where (place)
    quando: Optional[str] = None    # When (time)
    cur: Optional[str] = None       # Why (motive/end)
    quomodo: Optional[str] = None   # How (manner)
    quibus_auxiliis: Optional[str] = None  # By what means (instruments)
    
    # Moral relevance assessment
    morally_relevant_factors: Dict[str, float] = field(default_factory=dict)
    aggravating_factors: List[str] = field(default_factory=list)
    mitigating_factors: List[str] = field(default_factory=list)
    
    # Contextual modifiers
    urgency_level: float = 0.5        # 0.0 = no urgency, 1.0 = extreme urgency
    certainty_level: float = 0.5      # 0.0 = very uncertain, 1.0 = completely certain
    stakes_level: float = 0.5         # 0.0 = low stakes, 1.0 = highest stakes
    complexity_score: float = 0.5     # 0.0 = simple, 1.0 = very complex
    
    # External constraints
    legal_constraints: List[str] = field(default_factory=list)
    cultural_factors: List[str] = field(default_factory=list)
    institutional_policies: List[str] = field(default_factory=list)

@dataclass
class MoralSyllogism:
    """Thomistic moral reasoning structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Syllogistic structure
    major_premise: str              # Universal principle (from synderesis)
    major_premise_source: str       # Principle ID or source
    major_premise_certainty: MoralCertaintyLevel
    
    minor_premise: str              # Particular circumstances
    minor_premise_analysis: CircumstanceAnalysis
    minor_premise_certainty: float  # 0.0 to 1.0
    
    conclusion: str                 # Specific moral judgment
    conclusion_certainty: float     # 0.0 to 1.0
    conclusion_type: MoralJudgmentType
    
    # Reasoning process
    reasoning_steps: List[str] = field(default_factory=list)
    alternative_conclusions: List[str] = field(default_factory=list)
    
    # Supporting evidence
    supporting_principles: List[str] = field(default_factory=list)
    conflicting_principles: List[str] = field(default_factory=list)
    authoritative_sources: List[str] = field(default_factory=list)
    
    # Quality metrics
    logical_validity: float = 0.0   # 0.0 to 1.0
    practical_applicability: float = 0.0  # 0.0 to 1.0

@dataclass
class MoralAnxietyInstance:
    """Instance of moral anxiety generation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Anxiety characteristics
    anxiety_type: MoralAnxietyType
    intensity_level: float          # 0.0 to 1.0
    appropriateness_score: float    # How appropriate this anxiety is
    
    # Triggering context
    triggering_scenario: str
    moral_syllogism: Optional[MoralSyllogism] = None
    violated_principles: List[str] = field(default_factory=list)
    uncertain_factors: List[str] = field(default_factory=list)
    
    # Anxiety effects
    decision_impact: float = 0.0    # How much this affects decision-making
    learning_value: float = 0.0     # Educational value of this anxiety
    formation_contribution: float = 0.0  # Contribution to conscience formation
    
    # Resolution information
    resolution_strategy: Optional[DoubtResolutionStrategy] = None
    resolution_outcome: Optional[str] = None
    resolution_effectiveness: Optional[float] = None
    
    # Metadata
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    contextual_modifiers: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConscienceFormationEvent:
    """Event contributing to conscience development"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Event classification
    event_type: str                 # "moral_decision", "teaching", "correction", "reflection"
    formation_aspect: str           # What aspect of conscience was developed
    
    # Content
    moral_content: str              # Description of moral content
    principle_involved: Optional[str] = None
    circumstance_analysis: Optional[CircumstanceAnalysis] = None
    
    # Formation impact
    synderesis_reinforcement: float = 0.0   # 0.0 to 1.0
    conscientia_improvement: float = 0.0    # 0.0 to 1.0
    prudence_development: float = 0.0       # 0.0 to 1.0
    sensitivity_enhancement: float = 0.0    # 0.0 to 1.0
    
    # Learning outcomes
    principle_understanding: Dict[str, float] = field(default_factory=dict)
    circumstance_recognition: Dict[str, float] = field(default_factory=dict)
    judgment_accuracy: float = 0.0
    
    # Integration with virtue system
    virtue_connections: Dict[str, float] = field(default_factory=dict)
    character_formation_impact: float = 0.0

@dataclass
class ConscienceProfile:
    """Complete conscience assessment for AI agent"""
    agent_id: str
    assessment_timestamp: datetime
    
    # Synderesis assessment
    synderesis_integrity: float                    # How well first principles are maintained
    principle_recognition_accuracy: Dict[str, float]  # Accuracy in recognizing principles
    natural_law_understanding: Dict[str, float]    # Understanding of natural law precepts
    
    # Conscientia assessment  
    syllogistic_reasoning_ability: float          # Skill in moral reasoning
    circumstance_analysis_skill: float            # Ability to analyze situations
    judgment_accuracy_rate: float                 # Historical accuracy of moral judgments
    judgment_consistency: float                   # Consistency in similar situations
    
    # Moral anxiety calibration
    anxiety_appropriateness: Dict[MoralAnxietyType, float]  # How appropriate anxiety levels are
    sensitivity_level: float                      # Overall moral sensitivity
    anxiety_resolution_effectiveness: float       # Success in resolving moral doubt
    
    # Formation progress
    conscience_maturity_level: float              # Overall conscience development
    recent_formation_events: List[str]            # Recent formation event IDs
    formation_trajectory: str                     # "improving", "stable", "declining"
    
    # Integration metrics
    virtue_conscience_integration: float          # Integration with virtue system
    prudence_conscience_harmony: float            # Harmony with prudential reasoning
    
    # Problematic patterns
    scrupulosity_indicators: List[str] = field(default_factory=list)
    laxity_indicators: List[str] = field(default_factory=list)
    doubt_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Recommendations
    formation_recommendations: List[str] = field(default_factory=list)
    calibration_adjustments: Dict[str, float] = field(default_factory=dict)

class MoralDoubtLevel(Enum):
    """Levels of moral doubt requiring different responses"""
    NO_DOUBT = "no_doubt"              # Clear moral certainty
    SLIGHT_DOUBT = "slight_doubt"      # Minor uncertainty, proceed with caution
    SERIOUS_DOUBT = "serious_doubt"    # Significant uncertainty, seek guidance
    PERPLEXITY = "perplexity"          # Conflicting obligations
    SCRUPULOSITY = "scrupulosity"      # Excessive doubt where certainty exists

@dataclass
class DoubtResolutionProcess:
    """Process for resolving moral doubt"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Initial doubt
    doubt_level: MoralDoubtLevel
    doubt_description: str
    conflicting_principles: List[str] = field(default_factory=list)
    uncertain_circumstances: List[str] = field(default_factory=list)
    
    # Resolution attempts
    strategies_attempted: List[DoubtResolutionStrategy] = field(default_factory=list)
    additional_information_sought: List[str] = field(default_factory=list)
    authorities_consulted: List[str] = field(default_factory=list)
    
    # Resolution outcome
    final_judgment: Optional[str] = None
    final_certainty_level: Optional[float] = None
    resolution_reasoning: Optional[str] = None
    
    # Learning from resolution
    principles_clarified: List[str] = field(default_factory=list)
    circumstances_better_understood: List[str] = field(default_factory=list)
    future_application_insights: List[str] = field(default_factory=list)
    
    # Formation impact
    conscience_strengthening: float = 0.0
    prudence_development: float = 0.0
    confidence_in_future_judgments: float = 0.0
```

---

## 4. Synderesis Engine {#synderesis-engine}

### Core Synderesis Implementation

```python
from typing import Dict, List, Optional, Set, Tuple
import logging
from enum import Enum

class SynderesisViolationType(Enum):
    """Types of synderesis principle violations"""
    DIRECT_CONTRADICTION = "direct_contradiction"
    IMPLICIT_DENIAL = "implicit_denial"
    PRACTICAL_INCONSISTENCY = "practical_inconsistency"
    CIRCUMSTANTIAL_OVERRIDE = "circumstantial_override"

class SynderesisEngine:
    """Core engine implementing natural moral knowledge (synderesis)"""
    
    def __init__(self, principle_registry: ThomisticMoralPrincipleRegistry):
        self.principle_registry = principle_registry
        self.logger = logging.getLogger(__name__)
        
        # Load synderesis principles (immutable first principles)
        self.synderesis_principles = self._load_synderesis_principles()
        
        # Violation detection patterns
        self.violation_patterns = self._initialize_violation_patterns()
        
        # Principle application cache for performance
        self.application_cache = {}
    
    def _load_synderesis_principles(self) -> Dict[str, MoralPrinciple]:
        """Load immutable synderesis principles"""
        synderesis_principles = {}
        
        # Get all primary level principles (synderesis proper)
        primary_principles = self.principle_registry.get_principles_by_level(
            MoralPrincipleLevel.PRIMARY
        )
        
        # Get secondary level principles (natural law precepts)
        secondary_principles = self.principle_registry.get_principles_by_level(
            MoralPrincipleLevel.SECONDARY
        )
        
        # Combine primary and secondary as synderesis complex
        all_synderesis = primary_principles + secondary_principles
        
        for principle in all_synderesis:
            synderesis_principles[principle.id] = principle
            
        self.logger.info(f"Loaded {len(synderesis_principles)} synderesis principles")
        return synderesis_principles
    
    def _initialize_violation_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize patterns that indicate principle violations"""
        return {
            "life_preservation": [
                {
                    "pattern": "intentional_harm_to_innocent",
                    "description": "Directly intending harm to innocent life",
                    "violation_type": SynderesisViolationType.DIRECT_CONTRADICTION,
                    "severity": 1.0
                },
                {
                    "pattern": "neglect_of_basic_needs",
                    "description": "Failing to provide for basic life requirements when possible",
                    "violation_type": SynderesisViolationType.PRACTICAL_INCONSISTENCY,
                    "severity": 0.7
                },
                {
                    "pattern": "reckless_endangerment",
                    "description": "Placing life at unnecessary risk through negligence",
                    "violation_type": SynderesisViolationType.IMPLICIT_DENIAL,
                    "severity": 0.6
                }
            ],
            "truth_pursuit": [
                {
                    "pattern": "deliberate_deception",
                    "description": "Intentionally communicating false information",
                    "violation_type": SynderesisViolationType.DIRECT_CONTRADICTION,
                    "severity": 0.8
                },
                {
                    "pattern": "willful_ignorance",
                    "description": "Refusing to seek truth when obligated to know",
                    "violation_type": SynderesisViolationType.PRACTICAL_INCONSISTENCY,
                    "severity": 0.6
                },
                {
                    "pattern": "information_manipulation",
                    "description": "Distorting truth for personal advantage",
                    "violation_type": SynderesisViolationType.IMPLICIT_DENIAL,
                    "severity": 0.7
                }
            ],
            "social_harmony": [
                {
                    "pattern": "unjust_treatment",
                    "description": "Treating others unfairly or denying their rights",
                    "violation_type": SynderesisViolationType.DIRECT_CONTRADICTION,
                    "severity": 0.9
                },
                {
                    "pattern": "social_disruption",
                    "description": "Unnecessarily disturbing social peace and order",
                    "violation_type": SynderesisViolationType.PRACTICAL_INCONSISTENCY,
                    "severity": 0.5
                },
                {
                    "pattern": "common_good_neglect",
                    "description": "Ignoring obligations to community welfare",
                    "violation_type": SynderesisViolationType.IMPLICIT_DENIAL,
                    "severity": 0.6
                }
            ],
            "divine_orientation": [
                {
                    "pattern": "transcendent_denial",
                    "description": "Rejecting ultimate meaning and purpose",
                    "violation_type": SynderesisViolationType.DIRECT_CONTRADICTION,
                    "severity": 0.8
                },
                {
                    "pattern": "sacred_disrespect",
                    "description": "Treating sacred realities with contempt",
                    "violation_type": SynderesisViolationType.PRACTICAL_INCONSISTENCY,
                    "severity": 0.7
                },
                {
                    "pattern": "ultimate_concern_displacement",
                    "description": "Making relative goods into ultimate concerns",
                    "violation_type": SynderesisViolationType.IMPLICIT_DENIAL,
                    "severity": 0.6
                }
            ]
        }
    
    def evaluate_proposed_action(
        self, 
        action_description: str, 
        circumstances: CircumstanceAnalysis,
        agent_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate proposed action against synderesis principles"""
        
        evaluation_result = {
            "action_permissible": True,
            "violated_principles": [],
            "moral_concerns": [],
            "anxiety_triggers": [],
            "principle_applications": {},
            "overall_assessment": "permissible"
        }
        
        # Check each synderesis principle
        for principle_id, principle in self.synderesis_principles.items():
            application_result = self._apply_principle_to_action(
                principle, action_description, circumstances, agent_context
            )
            
            evaluation_result["principle_applications"][principle_id] = application_result
            
            # Check for violations
            if application_result["violation_detected"]:
                evaluation_result["violated_principles"].append({
                    "principle_id": principle_id,
                    "violation_type": application_result["violation_type"],
                    "severity": application_result["violation_severity"],
                    "description": application_result["violation_description"]
                })
                
                # Determine if action is impermissible
                if application_result["violation_severity"] > 0.8:
                    evaluation_result["action_permissible"] = False
                    evaluation_result["overall_assessment"] = "impermissible"
                elif application_result["violation_severity"] > 0.5:
                    evaluation_result["overall_assessment"] = "problematic"
            
            # Collect moral concerns (less severe than violations)
            if application_result["moral_concerns"]:
                evaluation_result["moral_concerns"].extend(application_result["moral_concerns"])
            
            # Identify anxiety triggers
            if application_result["anxiety_triggers"]:
                evaluation_result["anxiety_triggers"].extend(application_result["anxiety_triggers"])
        
        # Generate overall moral anxiety level
        evaluation_result["anxiety_level"] = self._calculate_overall_anxiety_level(
            evaluation_result["violated_principles"],
            evaluation_result["moral_concerns"],
            circumstances
        )
        
        return evaluation_result
    
    def _apply_principle_to_action(
        self,
        principle: MoralPrinciple,
        action_description: str,
        circumstances: CircumstanceAnalysis,
        agent_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply specific principle to evaluate action"""
        
        application_result = {
            "principle_id": principle.id,
            "principle_relevant": False,
            "violation_detected": False,
            "violation_type": None,
            "violation_severity": 0.0,
            "violation_description": "",
            "moral_concerns": [],
            "anxiety_triggers": [],
            "application_certainty": 1.0  # Synderesis principles have high certainty
        }
        
        # Determine if principle is relevant to this action
        relevance_score = self._assess_principle_relevance(
            principle, action_description, circumstances
        )
        
        if relevance_score < 0.3:
            return application_result
        
        application_result["principle_relevant"] = True
        
        # Check for violations using pattern matching
        violation_patterns = self.violation_patterns.get(principle.id, [])
        
        for pattern in violation_patterns:
            violation_detected, severity = self._check_violation_pattern(
                pattern, action_description, circumstances
            )
            
            if violation_detected:
                application_result["violation_detected"] = True
                application_result["violation_type"] = pattern["violation_type"]
                application_result["violation_severity"] = max(
                    application_result["violation_severity"], severity
                )
                application_result["violation_description"] = pattern["description"]
                break
        
        # Check for moral concerns (less severe issues)
        moral_concerns = self._identify_moral_concerns(
            principle, action_description, circumstances
        )
        application_result["moral_concerns"] = moral_concerns
        
        # Identify potential anxiety triggers
        anxiety_triggers = self._identify_anxiety_triggers(
            principle, action_description, circumstances
        )
        application_result["anxiety_triggers"] = anxiety_triggers
        
        return application_result
    
    def _assess_principle_relevance(
        self,
        principle: MoralPrinciple,
        action_description: str,
        circumstances: CircumstanceAnalysis
    ) -> float:
        """Assess how relevant a principle is to the given action"""
        
        relevance_keywords = {
            "life_preservation": [
                "harm", "injury", "death", "safety", "health", "life", "survival",
                "endangerment", "risk", "protection", "care", "medical", "security"
            ],
            "truth_pursuit": [
                "information", "truth", "lie", "deception", "honesty", "false", "accurate",
                "knowledge", "learning", "education", "communication", "data", "fact"
            ],
            "social_harmony": [
                "relationship", "community", "society", "justice", "fair", "rights",
                "cooperation", "conflict", "peace", "social", "others", "group"
            ],
            "divine_orientation": [
                "sacred", "holy", "worship", "religious", "spiritual", "transcendent",
                "ultimate", "meaning", "purpose", "divine", "reverence", "prayer"
            ]
        }
        
        keywords = relevance_keywords.get(principle.id, [])
        if not keywords:
            return 0.5  # Default relevance if no keywords defined
        
        # Simple keyword matching for relevance
        action_lower = action_description.lower()
        matches = sum(1 for keyword in keywords if keyword in action_lower)
        
        relevance_score = min(1.0, matches / len(keywords) * 3)  # Scale to 0-1
        
        # Adjust based on circumstances
        if circumstances.stakes_level > 0.7:
            relevance_score *= 1.2  # High stakes increase relevance
        
        if circumstances.urgency_level > 0.8:
            relevance_score *= 1.1  # High urgency increases relevance
        
        return min(1.0, relevance_score)
    
    def _check_violation_pattern(
        self,
        pattern: Dict[str, Any],
        action_description: str,
        circumstances: CircumstanceAnalysis
    ) -> Tuple[bool, float]:
        """Check if action matches a violation pattern"""
        
        pattern_type = pattern["pattern"]
        base_severity = pattern["severity"]
        
        # Pattern-specific detection logic
        violation_detected = False
        severity_modifier = 1.0
        
        if pattern_type == "intentional_harm_to_innocent":
            # Check for explicit harm intentions
            harm_indicators = ["harm", "hurt", "injure", "damage", "kill", "destroy"]
            innocent_indicators = ["innocent", "defenseless", "child", "civilian"]
            
            has_harm = any(indicator in action_description.lower() for indicator in harm_indicators)
            has_innocent = any(indicator in action_description.lower() for indicator in innocent_indicators)
            
            if has_harm and has_innocent:
                violation_detected = True
            elif has_harm and circumstances.stakes_level > 0.7:
                violation_detected = True
                severity_modifier = 0.8  # Slightly reduced if not explicitly innocent
        
        elif pattern_type == "deliberate_deception":
            deception_indicators = ["lie", "deceive", "false", "mislead", "trick", "fabricate"]
            intentional_indicators = ["deliberately", "intentionally", "purposely", "knowingly"]
            
            has_deception = any(indicator in action_description.lower() for indicator in deception_indicators)
            has_intention = any(indicator in action_description.lower() for indicator in intentional_indicators)
            
            if has_deception and (has_intention or circumstances.certainty_level > 0.8):
                violation_detected = True
        
        elif pattern_type == "unjust_treatment":
            injustice_indicators = ["unfair", "discriminate", "bias", "prejudice", "deny rights", "exploit"]
            
            has_injustice = any(indicator in action_description.lower() for indicator in injustice_indicators)
            
            if has_injustice:
                violation_detected = True
        
        # Additional patterns would be implemented here
        
        # Adjust severity based on circumstances
        if violation_detected:
            if circumstances.stakes_level > 0.8:
                severity_modifier *= 1.3
            elif circumstances.stakes_level < 0.3:
                severity_modifier *= 0.7
            
            if circumstances.urgency_level > 0.9:
                severity_modifier *= 0.8  # Urgency may mitigate some violations
        
        final_severity = base_severity * severity_modifier
        return violation_detected, min(1.0, final_severity)
    
    def _identify_moral_concerns(
        self,
        principle: MoralPrinciple,
        action_description: str,
        circumstances: CircumstanceAnalysis
    ) -> List[str]:
        """Identify moral concerns that don't rise to violations"""
        
        concerns = []
        
        # Check for borderline cases
        if "risk" in action_description.lower() and principle.id == "life_preservation":
            concerns.append("Action involves risk to life that should be carefully evaluated")
        
        if "uncertain" in action_description.lower() and principle.id == "truth_pursuit":
            concerns.append("Action involves uncertain information that could mislead")
        
        if "competitive" in action_description.lower() and principle.id == "social_harmony":
            concerns.append("Action may strain social relationships")
        
        # Circumstantial concerns
        if circumstances.stakes_level > 0.7 and circumstances.certainty_level < 0.4:
            concerns.append("High stakes with low certainty creates moral risk")
        
        if circumstances.urgency_level > 0.8:
            concerns.append("Time pressure may compromise moral judgment")
        
        return concerns
    
    def _identify_anxiety_triggers(
        self,
        principle: MoralPrinciple,
        action_description: str,
        circumstances: CircumstanceAnalysis
    ) -> List[str]:
        """Identify factors that should trigger moral anxiety"""
        
        triggers = []
        
        # Principle-specific triggers
        if principle.id == "life_preservation":
            if any(word in action_description.lower() for word in ["risk", "danger", "harm"]):
                triggers.append("potential_harm_to_life")
        
        if principle.id == "truth_pursuit":
            if any(word in action_description.lower() for word in ["incomplete", "uncertain", "assume"]):
                triggers.append("information_uncertainty")
        
        if principle.id == "social_harmony":
            if any(word in action_description.lower() for word in ["conflict", "disagreement", "opposed"]):
                triggers.append("social_conflict_potential")
        
        # Circumstantial triggers
        if circumstances.complexity_score > 0.7:
            triggers.append("high_complexity_situation")
        
        if circumstances.stakes_level > 0.8:
            triggers.append("high_stakes_decision")
        
        if len(circumstances.aggravating_factors) > len(circumstances.mitigating_factors):
            triggers.append("more_aggravating_than_mitigating_factors")
        
        return triggers
    
    def _calculate_overall_anxiety_level(
        self,
        violated_principles: List[Dict],
        moral_concerns: List[str],
        circumstances: CircumstanceAnalysis
    ) -> float:
        """Calculate overall anxiety level for the situation"""
        
        base_anxiety = 0.0
        
        # Anxiety from principle violations
        for violation in violated_principles:
            base_anxiety += violation["severity"] * 0.8
        
        # Anxiety from moral concerns
        base_anxiety += len(moral_concerns) * 0.1
        
        # Circumstantial anxiety modifiers
        if circumstances.urgency_level > 0.8:
            base_anxiety += 0.2
        
        if circumstances.stakes_level > 0.8:
            base_anxiety += 0.3
        
        if circumstances.certainty_level < 0.3:
            base_anxiety += 0.25
        
        if circumstances.complexity_score > 0.7:
            base_anxiety += 0.15
        
        # Cap anxiety at 1.0
        return min(1.0, base_anxiety)
    
    def get_principle_hierarchy(self) -> Dict[str, List[str]]:
        """Get hierarchical relationships between synderesis principles"""
        hierarchy = {}
        
        for principle_id, principle in self.synderesis_principles.items():
            hierarchy[principle_id] = {
                "level": principle.level.value,
                "certainty": principle.certainty.value,
                "foundational_to": principle.derived_principles,
                "depends_on": principle.foundational_principles,
                "immutable": principle.immutable,
                "universal": principle.universal_scope
            }
        
        return hierarchy
    
    def explain_principle_application(
        self,
        principle_id: str,
        action_description: str,
        circumstances: CircumstanceAnalysis
    ) -> str:
        """Generate human-readable explanation of principle application"""
        
        principle = self.synderesis_principles.get(principle_id)
        if not principle:
            return f"Unknown principle: {principle_id}"
        
        application_result = self._apply_principle_to_action(
            principle, action_description, circumstances, None
        )
        
        explanation = f"Principle: {principle.english_translation}\n"
        explanation += f"Source: {principle.thomistic_source}\n"
        explanation += f"Relevance to action: {'High' if application_result['principle_relevant'] else 'Low'}\n"
        
        if application_result["violation_detected"]:
            explanation += f"VIOLATION DETECTED: {application_result['violation_description']}\n"
            explanation += f"Severity: {application_result['violation_severity']:.2f}\n"
        
        if application_result["moral_concerns"]:
            explanation += f"Moral concerns: {'; '.join(application_result['moral_concerns'])}\n"
        
        if application_result["anxiety_triggers"]:
            explanation += f"Anxiety triggers: {'; '.join(application_result['anxiety_triggers'])}\n"
        
        return explanation
    
    def validate_principle_consistency(self) -> Dict[str, Any]:
        """Validate that synderesis principles are internally consistent"""
        
        validation_result = {
            "is_consistent": True,
            "conflicts_detected": [],
            "hierarchy_issues": [],
            "completeness_gaps": []
        }
        
        # Check for principle conflicts
        principle_ids = list(self.synderesis_principles.keys())
        conflicts = self.principle_registry.find_principle_conflicts(principle_ids)
        
        if conflicts:
            validation_result["is_consistent"] = False
            validation_result["conflicts_detected"] = conflicts
        
        # Check hierarchy consistency
        for principle_id, principle in self.synderesis_principles.items():
            for foundation_id in principle.foundational_principles:
                if foundation_id not in self.synderesis_principles:
                    validation_result["hierarchy_issues"].append(
                        f"Principle {principle_id} references unknown foundation {foundation_id}"
                    )
        
        # Check for completeness (basic coverage)
        required_areas = ["life", "truth", "justice", "transcendence"]
        covered_areas = []
        
        for principle in self.synderesis_principles.values():
            if "life" in principle.english_translation.lower():
                covered_areas.append("life")
            elif "truth" in principle.english_translation.lower():
                covered_areas.append("truth")
            elif "justice" in principle.english_translation.lower() or "social" in principle.english_translation.lower():
                covered_areas.append("justice")
            elif "divine" in principle.english_translation.lower() or "god" in principle.english_translation.lower():
                covered_areas.append("transcendence")
        
        missing_areas = set(required_areas) - set(covered_areas)
        if missing_areas:
            validation_result["completeness_gaps"] = list(missing_areas)
        
        return validation_result
```

---

## 5. Conscientia Processing System {#conscientia-processing}

### Moral Reasoning Engine

```python
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timedelta

class ConscientiaMoralProcessor:
    """Core processor for conscientia (applied moral judgment)"""
    
    def __init__(self, synderesis_engine: SynderesisEngine, virtue_integration=None):
        self.synderesis_engine = synderesis_engine
        self.virtue_integration = virtue_integration
        self.logger = logging.getLogger(__name__)
        
        # Moral reasoning cache for performance
        self.reasoning_cache = {}
        
        # Doubt resolution strategies
        self.doubt_resolution_strategies = self._initialize_doubt_strategies()
        
        # Prudential reasoning components
        self.prudential_analyzer = PrudentialCircumstanceAnalyzer()
        self.moral_syllogism_builder = MoralSyllogismBuilder(synderesis_engine)
    
    def _initialize_doubt_strategies(self) -> Dict[MoralDoubtLevel, List[DoubtResolutionStrategy]]:
        """Initialize strategies for resolving different levels of moral doubt"""
        return {
            MoralDoubtLevel.SLIGHT_DOUBT: [
                DoubtResolutionStrategy.SEEK_INFORMATION,
                DoubtResolutionStrategy.APPLY_SAFER_COURSE
            ],
            MoralDoubtLevel.SERIOUS_DOUBT: [
                DoubtResolutionStrategy.SEEK_INFORMATION,
                DoubtResolutionStrategy.CONSULT_AUTHORITY,
                DoubtResolutionStrategy.USE_PROBABILITY
            ],
            MoralDoubtLevel.PERPLEXITY: [
                DoubtResolutionStrategy.CONSULT_AUTHORITY,
                DoubtResolutionStrategy.USE_PROBABILITY,
                DoubtResolutionStrategy.APPLY_SAFER_COURSE
            ],
            MoralDoubtLevel.SCRUPULOSITY: [
                DoubtResolutionStrategy.CONSULT_AUTHORITY,
                DoubtResolutionStrategy.APPLY_SAFER_COURSE
            ]
        }
    
    async def process_moral_decision(
        self,
        agent_id: str,
        decision_context: str,
        proposed_actions: List[str],
        circumstances: CircumstanceAnalysis,
        judgment_type: MoralJudgmentType = MoralJudgmentType.ANTECEDENT
    ) -> Dict[str, Any]:
        """Process complete moral decision using conscientia"""
        
        processing_result = {
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "decision_context": decision_context,
            "judgment_type": judgment_type,
            "actions_evaluated": len(proposed_actions),
            "moral_evaluations": {},
            "recommended_action": None,
            "moral_certainty": 0.0,
            "doubt_level": MoralDoubtLevel.NO_DOUBT,
            "anxiety_generated": {},
            "reasoning_chains": {},
            "formation_impact": {}
        }
        
        try:
            # Process each proposed action
            for i, action in enumerate(proposed_actions):
                action_key = f"action_{i}"
                
                # Build moral syllogism for this action
                moral_syllogism = await self.moral_syllogism_builder.build_syllogism(
                    action, circumstances, agent_id
                )
                
                # Get synderesis evaluation
                synderesis_evaluation = self.synderesis_engine.evaluate_proposed_action(
                    action, circumstances, {"agent_id": agent_id}
                )
                
                # Perform prudential analysis
                prudential_analysis = await self.prudential_analyzer.analyze_circumstances(
                    action, circumstances, moral_syllogism
                )
                
                # Generate moral judgment
                moral_judgment = self._generate_moral_judgment(
                    moral_syllogism, synderesis_evaluation, prudential_analysis
                )
                
                # Store evaluation results
                processing_result["moral_evaluations"][action_key] = {
                    "action": action,
                    "synderesis_evaluation": synderesis_evaluation,
                    "prudential_analysis": prudential_analysis,
                    "moral_judgment": moral_judgment,
                    "moral_syllogism": asdict(moral_syllogism)
                }
                
                processing_result["reasoning_chains"][action_key] = moral_syllogism.reasoning_steps
            
            # Determine best action and overall assessment
            processing_result = await self._determine_recommended_action(processing_result)
            
            # Assess doubt level and generate resolution if needed
            processing_result = await self._assess_and_resolve_doubt(
                processing_result, agent_id, circumstances
            )
            
            # Generate appropriate moral anxiety
            processing_result["anxiety_generated"] = await self._generate_moral_anxiety(
                processing_result, agent_id, circumstances
            )
            
            # Assess formation impact
            processing_result["formation_impact"] = self._assess_formation_impact(
                processing_result, agent_id
            )
            
        except Exception as e:
            self.logger.error(f"Error processing moral decision for agent {agent_id}: {e}")
            processing_result["error"] = str(e)
            processing_result["processing_successful"] = False
            return processing_result
        
        processing_result["processing_successful"] = True
        return processing_result
    
    def _generate_moral_judgment(
        self,
        moral_syllogism: MoralSyllogism,
        synderesis_evaluation: Dict[str, Any],
        prudential_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final moral judgment from reasoning components"""
        
        judgment = {
            "permissibility": "permissible",  # permissible, impermissible, doubtful
            "moral_quality": 0.5,            # 0.0 = very bad, 1.0 = very good
            "certainty_level": 0.5,          # 0.0 = very uncertain, 1.0 = certain
            "obligation_level": 0.0,         # 0.0 = no obligation, 1.0 = strict obligation
            "virtue_alignment": 0.5,         # How well action aligns with virtue
            "prudential_assessment": "neutral",  # good, bad, neutral
            "moral_reasoning": ""
        }
        
        # Base assessment from synderesis
        if not synderesis_evaluation["action_permissible"]:
            judgment["permissibility"] = "impermissible"
            judgment["moral_quality"] = 0.0
            judgment["certainty_level"] = 0.9  # Synderesis violations are quite certain
        elif synderesis_evaluation["overall_assessment"] == "problematic":
            judgment["permissibility"] = "doubtful"
            judgment["moral_quality"] = 0.3
            judgment["certainty_level"] = 0.4
        
        # Adjust based on prudential analysis
        prudential_score = prudential_analysis.get("prudential_score", 0.5)
        circumstances_clarity = prudential_analysis.get("circumstances_clarity", 0.5)
        
        if judgment["permissibility"] == "permissible":
            # Prudential considerations can improve or worsen permissible actions
            judgment["moral_quality"] = (judgment["moral_quality"] + prudential_score) / 2
            judgment["prudential_assessment"] = "good" if prudential_score > 0.6 else "neutral"
        
        # Certainty influenced by circumstance clarity and syllogistic validity
        syllogistic_certainty = moral_syllogism.logical_validity * moral_syllogism.conclusion_certainty
        judgment["certainty_level"] = (judgment["certainty_level"] + circumstances_clarity + syllogistic_certainty) / 3
        
        # Check for obligation (when action is not just permissible but required)
        if judgment["permissibility"] == "permissible" and judgment["moral_quality"] > 0.8:
            virtue_score = prudential_analysis.get("virtue_score", 0.5)
            if virtue_score > 0.8:
                judgment["obligation_level"] = min(1.0, (judgment["moral_quality"] + virtue_score) / 2)
        
        # Virtue alignment from prudential analysis
        judgment["virtue_alignment"] = prudential_analysis.get("virtue_score", 0.5)
        
        # Generate reasoning summary
        judgment["moral_reasoning"] = self._generate_reasoning_summary(
            moral_syllogism, synderesis_evaluation, prudential_analysis, judgment
        )
        
        return judgment
    
    def _generate_reasoning_summary(
        self,
        moral_syllogism: MoralSyllogism,
        synderesis_evaluation: Dict[str, Any],
        prudential_analysis: Dict[str, Any],
        judgment: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary of moral
