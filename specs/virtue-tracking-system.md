 async def get_virtue_profile(self, agent_id: str) -> Optional[VirtueProfile]:
        """Get cached virtue profile"""
        try:
            cache_key = self._generate_cache_key(self.PROFILE_KEY_PREFIX, agent_id)
            
            cached_value = await self.redis.get(cache_key)
            if cached_value:
                profile_data = pickle.loads(cached_value)
                self.logger.debug(f"Cache hit for virtue profile: {cache_key}")
                
                # Reconstruct VirtueProfile object
                virtue_strengths = {VirtueType(k): v for k, v in profile_data["virtue_strengths"].items()}
                virtue_development_rates = None
                if profile_data.get("virtue_development_rates"):
                    virtue_development_rates = {VirtueType(k): v for k, v in profile_data["virtue_development_rates"].items()}
                
                return VirtueProfile(
                    agent_id=profile_data["agent_id"],
                    assessment_timestamp=profile_data["assessment_timestamp"],
                    virtue_strengths=virtue_strengths,
                    virtue_development_rates=virtue_development_rates,
                    virtue_interactions=profile_data.get("virtue_interactions"),
                    dominant_virtues=[VirtueType(v) for v in profile_data["dominant_virtues"]],
                    developing_virtues=[VirtueType(v) for v in profile_data["developing_virtues"]],
                    virtue_integration_score=profile_data["virtue_integration_score"],
                    moral_character_assessment=profile_data["moral_character_assessment"]
                )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache get profile failed: {e}")
            return None
    
    async def set_virtue_profile(self, profile: VirtueProfile, ttl: Optional[int] = None):
        """Cache virtue profile"""
        try:
            cache_key = self._generate_cache_key(self.PROFILE_KEY_PREFIX, profile.agent_id)
            
            # Serialize profile data
            cache_data = {
                "agent_id": profile.agent_id,
                "assessment_timestamp": profile.assessment_timestamp,
                "virtue_strengths": {k.value: v for k, v in profile.virtue_strengths.items()},
                "virtue_development_rates": {k.value: v for k, v in profile.virtue_development_rates.items()} if profile.virtue_development_rates else None,
                "virtue_interactions": profile.virtue_interactions,
                "dominant_virtues": [v.value for v in profile.dominant_virtues],
                "developing_virtues": [v.value for v in profile.developing_virtues],
                "virtue_integration_score": profile.virtue_integration_score,
                "moral_character_assessment": profile.moral_character_assessment,
                "cached_at": datetime.now()
            }
            
            await self.redis.setex(
                cache_key,
                ttl or self.default_ttl,
                pickle.dumps(cache_data)
            )
            
            self.logger.debug(f"Cached virtue profile: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Cache set profile failed: {e}")
    
    async def invalidate_agent_cache(self, agent_id: str):
        """Invalidate all cached data for an agent"""
        try:
            # Find and delete all keys related to this agent
            patterns = [
                f"{self.STRENGTH_KEY_PREFIX}*{agent_id}*",
                f"{self.PROFILE_KEY_PREFIX}*{agent_id}*",
                f"{self.SYNERGY_KEY_PREFIX}*{agent_id}*",
                f"{self.INSTANCE_KEY_PREFIX}*{agent_id}*"
            ]
            
            for pattern in patterns:
                async for key in self.redis.scan_iter(match=pattern):
                    await self.redis.delete(key)
            
            self.logger.info(f"Invalidated cache for agent: {agent_id}")
            
        except Exception as e:
            self.logger.warning(f"Cache invalidation failed: {e}")

class OptimizedVirtueStrengthCalculator(VirtueStrengthCalculator):
    """Performance-optimized version of virtue strength calculator"""
    
    def __init__(self, virtue_registry: ThomisticVirtueRegistry, cache: VirtueCalculationCache):
        super().__init__(virtue_registry)
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.batch_size = 100  # Process demonstrations in batches
    
    async def measure_virtue_strength_cached(
        self,
        virtue_type: VirtueType,
        agent_id: str,
        time_window_days: int = 30,
        force_recalculation: bool = False
    ) -> float:
        """Measure virtue strength with caching"""
        
        # Check cache first (unless forced recalculation)
        if not force_recalculation:
            cached_strength = await self.cache.get_virtue_strength(
                agent_id, virtue_type, time_window_days
            )
            if cached_strength is not None:
                return cached_strength
        
        # Get recent demonstrations (this would normally hit database)
        recent_demonstrations = await self._get_recent_demonstrations(
            agent_id, virtue_type, time_window_days
        )
        
        # Calculate strength
        calculated_strength = await self._calculate_strength_optimized(
            virtue_type, recent_demonstrations, time_window_days
        )
        
        # Cache result
        await self.cache.set_virtue_strength(
            agent_id, virtue_type, time_window_days, calculated_strength,
            metadata={"demonstration_count": len(recent_demonstrations)}
        )
        
        return calculated_strength
    
    async def _calculate_strength_optimized(
        self,
        virtue_type: VirtueType,
        demonstrations: List[VirtueInstance],
        time_window_days: int
    ) -> float:
        """Optimized strength calculation using async processing"""
        
        if not demonstrations:
            return 0.0
        
        # Process demonstrations in batches for better performance
        if len(demonstrations) > self.batch_size:
            return await self._calculate_strength_batched(
                virtue_type, demonstrations, time_window_days
            )
        else:
            return await self._calculate_strength_single_batch(
                virtue_type, demonstrations, time_window_days
            )
    
    async def _calculate_strength_batched(
        self,
        virtue_type: VirtueType,
        demonstrations: List[VirtueInstance],
        time_window_days: int
    ) -> float:
        """Calculate strength for large datasets using batching"""
        
        # Split demonstrations into batches
        batches = [
            demonstrations[i:i + self.batch_size] 
            for i in range(0, len(demonstrations), self.batch_size)
        ]
        
        # Process batches concurrently
        loop = asyncio.get_event_loop()
        
        batch_results = await asyncio.gather(*[
            loop.run_in_executor(
                self.executor,
                self._process_demonstration_batch,
                virtue_type,
                batch
            )
            for batch in batches
        ])
        
        # Combine batch results
        total_quality = sum(result["total_quality"] for result in batch_results)
        total_count = sum(result["count"] for result in batch_results)
        combined_factors = self._combine_batch_factors(batch_results)
        
        if total_count == 0:
            return 0.0
        
        # Calculate final strength
        base_strength = total_quality / total_count
        habit_multiplier = self.habit_strength_model.calculate_habit_strength(
            repetition_count=total_count,
            time_distribution=combined_factors["time_distribution"],
            consistency_score=combined_factors["consistency"]
        )
        
        difficulty_adjustment = combined_factors["difficulty_adjustment"]
        virtue_specific_adjustment = combined_factors["virtue_specific"]
        
        final_strength = min(1.0, base_strength * habit_multiplier * 
                           difficulty_adjustment * virtue_specific_adjustment)
        
        return final_strength
    
    def _process_demonstration_batch(
        self,
        virtue_type: VirtueType,
        batch: List[VirtueInstance]
    ) -> Dict[str, Any]:
        """Process a batch of demonstrations (CPU-intensive work)"""
        
        quality_scores = [demo.strength_measurement for demo in batch]
        total_quality = sum(quality_scores)
        
        # Calculate batch-specific factors
        time_distribution = self._analyze_time_distribution(batch)
        consistency = self._calculate_consistency(quality_scores)
        difficulty_adjustment = self._calculate_difficulty_adjustment(batch)
        virtue_specific = self._apply_virtue_specific_logic(virtue_type, batch)
        
        return {
            "total_quality": total_quality,
            "count": len(batch),
            "time_distribution": time_distribution,
            "consistency": consistency,
            "difficulty_adjustment": difficulty_adjustment,
            "virtue_specific": virtue_specific
        }
    
    def _combine_batch_factors(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine factors from multiple batches"""
        
        total_count = sum(result["count"] for result in batch_results)
        
        if total_count == 0:
            return {
                "time_distribution": {"regularity_score": 0.5, "recent_frequency": 0.5},
                "consistency": 0.5,
                "difficulty_adjustment": 1.0,
                "virtue_specific": 1.0
            }
        
        # Weighted average based on batch sizes
        combined_consistency = sum(
            result["consistency"] * result["count"] for result in batch_results
        ) / total_count
        
        combined_difficulty = sum(
            result["difficulty_adjustment"] * result["count"] for result in batch_results
        ) / total_count
        
        combined_virtue_specific = sum(
            result["virtue_specific"] * result["count"] for result in batch_results
        ) / total_count
        
        # Combine time distributions
        total_regularity = sum(
            result["time_distribution"]["regularity_score"] * result["count"] 
            for result in batch_results
        ) / total_count
        
        total_frequency = sum(
            result["time_distribution"]["recent_frequency"] * result["count"] 
            for result in batch_results
        ) / total_count
        
        return {
            "time_distribution": {
                "regularity_score": total_regularity,
                "recent_frequency": total_frequency
            },
            "consistency": combined_consistency,
            "difficulty_adjustment": combined_difficulty,
            "virtue_specific": combined_virtue_specific
        }
    
    async def _get_recent_demonstrations(
        self,
        agent_id: str,
        virtue_type: VirtueType,
        time_window_days: int
    ) -> List[VirtueInstance]:
        """Get recent demonstrations (would interface with database service)"""
        # This would be implemented to interface with the database service
        # Placeholder implementation
        return []

class VirtueCalculationOptimizer:
    """Optimizer for virtue calculation performance"""
    
    def __init__(self):
        self.calculation_metrics = {}
        self.performance_thresholds = {
            "strength_calculation_ms": 100,
            "profile_generation_ms": 500,
            "synergy_calculation_ms": 50
        }
        self.logger = logging.getLogger(__name__)
    
    def measure_calculation_performance(self, operation_name: str):
        """Decorator to measure calculation performance"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = asyncio.get_event_loop().time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    end_time = asyncio.get_event_loop().time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    # Record metrics
                    self._record_performance_metric(operation_name, duration_ms, True)
                    
                    # Check if performance is degrading
                    if duration_ms > self.performance_thresholds.get(operation_name, 1000):
                        self.logger.warning(
                            f"Performance degradation detected: {operation_name} took {duration_ms:.2f}ms"
                        )
                    
                    return result
                    
                except Exception as e:
                    end_time = asyncio.get_event_loop().time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    # Record failed operation
                    self._record_performance_metric(operation_name, duration_ms, False)
                    raise
            
            return wrapper
        return decorator
    
    def _record_performance_metric(self, operation_name: str, duration_ms: float, success: bool):
        """Record performance metric"""
        if operation_name not in self.calculation_metrics:
            self.calculation_metrics[operation_name] = {
                "total_operations": 0,
                "successful_operations": 0,
                "total_duration_ms": 0.0,
                "min_duration_ms": float('inf'),
                "max_duration_ms": 0.0,
                "recent_durations": []
            }
        
        metrics = self.calculation_metrics[operation_name]
        metrics["total_operations"] += 1
        if success:
            metrics["successful_operations"] += 1
        
        metrics["total_duration_ms"] += duration_ms
        metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration_ms)
        metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration_ms)
        
        # Keep recent durations for moving average (last 100 operations)
        metrics["recent_durations"].append(duration_ms)
        if len(metrics["recent_durations"]) > 100:
            metrics["recent_durations"].pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations"""
        summary = {}
        
        for operation_name, metrics in self.calculation_metrics.items():
            if metrics["total_operations"] > 0:
                avg_duration = metrics["total_duration_ms"] / metrics["total_operations"]
                success_rate = metrics["successful_operations"] / metrics["total_operations"]
                
                recent_avg = (
                    sum(metrics["recent_durations"]) / len(metrics["recent_durations"])
                    if metrics["recent_durations"] else 0
                )
                
                summary[operation_name] = {
                    "total_operations": metrics["total_operations"],
                    "success_rate": success_rate,
                    "avg_duration_ms": avg_duration,
                    "recent_avg_duration_ms": recent_avg,
                    "min_duration_ms": metrics["min_duration_ms"],
                    "max_duration_ms": metrics["max_duration_ms"],
                    "performance_status": "good" if recent_avg < self.performance_thresholds.get(operation_name, 1000) else "degraded"
                }
        
        return summary

class DatabaseOptimizationService:
    """Service for optimizing database operations"""
    
    def __init__(self, connection_pool):
        self.pool = connection_pool
        self.query_cache = {}
        self.logger = logging.getLogger(__name__)
    
    async def execute_optimized_query(
        self,
        query: str,
        params: tuple,
        cache_key: Optional[str] = None,
        cache_ttl: int = 300
    ) -> List[Dict]:
        """Execute database query with optimization"""
        
        # Check query cache if cache_key provided
        if cache_key and cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if datetime.now() - cache_entry["timestamp"] < timedelta(seconds=cache_ttl):
                self.logger.debug(f"Query cache hit: {cache_key}")
                return cache_entry["results"]
        
        # Execute query with connection pooling
        async with self.pool.acquire() as conn:
            try:
                # Use prepared statement for better performance
                stmt = await conn.prepare(query)
                results = await stmt.fetch(*params)
                
                # Convert to list of dicts
                result_list = [dict(row) for row in results]
                
                # Cache results if cache_key provided
                if cache_key:
                    self.query_cache[cache_key] = {
                        "results": result_list,
                        "timestamp": datetime.now()
                    }
                
                return result_list
                
            except Exception as e:
                self.logger.error(f"Database query failed: {e}")
                raise
    
    async def batch_insert_virtue_instances(
        self,
        virtue_instances: List[VirtueInstance],
        agent_id: str
    ):
        """Optimized batch insert for virtue instances"""
        
        if not virtue_instances:
            return
        
        async with self.pool.acquire() as conn:
            try:
                # Prepare batch insert query
                query = """
                INSERT INTO virtue_instances (
                    id, agent_id, virtue_type, demonstration_context, 
                    strength_measurement, confidence_level, timestamp,
                    contributing_factors, circumstantial_modifiers, validation_sources
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                # Prepare data for batch insert
                batch_data = []
                for instance in virtue_instances:
                    batch_data.append((
                        instance.id,
                        agent_id,
                        instance.virtue_type.value,
                        instance.demonstration_context,
                        instance.strength_measurement,
                        instance.confidence_level,
                        instance.timestamp,
                        json.dumps(instance.contributing_factors),
                        json.dumps(instance.circumstantial_modifiers),
                        instance.validation_sources
                    ))
                
                # Execute batch insert
                await conn.executemany(query, batch_data)
                
                self.logger.info(f"Batch inserted {len(virtue_instances)} virtue instances")
                
            except Exception as e:
                self.logger.error(f"Batch insert failed: {e}")
                raise
    
    async def get_aggregated_virtue_metrics(
        self,
        agent_id: str,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Get aggregated virtue metrics with optimized query"""
        
        cache_key = f"metrics:{agent_id}:{time_window_days}"
        
        query = """
        SELECT 
            virtue_type,
            COUNT(*) as instance_count,
            AVG(strength_measurement) as avg_strength,
            STDDEV(strength_measurement) as strength_stddev,
            MIN(strength_measurement) as min_strength,
            MAX(strength_measurement) as max_strength,
            AVG(confidence_level) as avg_confidence
        FROM virtue_instances 
        WHERE agent_id = $1 
        AND timestamp >= $2
        GROUP BY virtue_type
        """
        
        start_date = datetime.now() - timedelta(days=time_window_days)
        results = await self.execute_optimized_query(
            query, (agent_id, start_date), cache_key, cache_ttl=600
        )
        
        # Convert to more useful format
        metrics = {}
        for row in results:
            metrics[row["virtue_type"]] = {
                "instance_count": row["instance_count"],
                "avg_strength": float(row["avg_strength"]),
                "strength_stddev": float(row["strength_stddev"] or 0),
                "min_strength": float(row["min_strength"]),
                "max_strength": float(row["max_strength"]),
                "avg_confidence": float(row["avg_confidence"])
            }
        
        return metrics
    
    async def cleanup_old_data(self, retention_days: int = 365):
        """Clean up old data to maintain database performance"""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        async with self.pool.acquire() as conn:
            try:
                # Clean up old virtue instances
                deleted_instances = await conn.execute(
                    "DELETE FROM virtue_instances WHERE timestamp < $1",
                    cutoff_date
                )
                
                # Clean up old strength calculations
                deleted_calculations = await conn.execute(
                    "DELETE FROM virtue_strength_calculations WHERE calculation_timestamp < $1",
                    cutoff_date
                )
                
                # Clean up old profiles (keep at least one per agent)
                await conn.execute("""
                    DELETE FROM virtue_profiles 
                    WHERE assessment_timestamp < $1 
                    AND id NOT IN (
                        SELECT DISTINCT ON (agent_id) id 
                        FROM virtue_profiles 
                        ORDER BY agent_id, assessment_timestamp DESC
                    )
                """, cutoff_date)
                
                self.logger.info(
                    f"Cleanup completed: {deleted_instances} instances, "
                    f"{deleted_calculations} calculations removed"
                )
                
            except Exception as e:
                self.logger.error(f"Data cleanup failed: {e}")
                raise

class VirtueCalculationPool:
    """Pool for managing concurrent virtue calculations"""
    
    def __init__(self, max_workers: int = None, queue_size: int = 1000):
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.calculation_queue = asyncio.Queue(maxsize=queue_size)
        self.active_calculations = {}
        self.logger = logging.getLogger(__name__)
        self.performance_metrics = VirtueCalculationOptimizer()
    
    async def submit_calculation(
        self,
        calculation_id: str,
        calculation_func: callable,
        *args,
        **kwargs
    ) -> asyncio.Future:
        """Submit calculation to the pool"""
        
        if calculation_id in self.active_calculations:
            # Return existing calculation if already running
            return self.active_calculations[calculation_id]
        
        # Create future for this calculation
        future = asyncio.get_event_loop().create_future()
        self.active_calculations[calculation_id] = future
        
        # Add to queue
        await self.calculation_queue.put({
            "id": calculation_id,
            "func": calculation_func,
            "args": args,
            "kwargs": kwargs,
            "future": future
        })
        
        return future
    
    async def start_workers(self):
        """Start worker tasks to process calculations"""
        workers = []
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} calculation workers")
        return workers
    
    async def _worker(self, worker_name: str):
        """Worker task to process calculation queue"""
        while True:
            try:
                # Get calculation from queue
                calculation = await self.calculation_queue.get()
                calculation_id = calculation["id"]
                
                self.logger.debug(f"{worker_name} processing {calculation_id}")
                
                try:
                    # Execute calculation in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        calculation["func"],
                        *calculation["args"],
                        **calculation["kwargs"]
                    )
                    
                    # Set result on future
                    calculation["future"].set_result(result)
                    
                except Exception as e:
                    # Set exception on future
                    calculation["future"].set_exception(e)
                    self.logger.error(f"Calculation {calculation_id} failed: {e}")
                
                finally:
                    # Remove from active calculations
                    self.active_calculations.pop(calculation_id, None)
                    self.calculation_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def get_calculation_status(self) -> Dict[str, Any]:
        """Get status of calculation pool"""
        return {
            "max_workers": self.max_workers,
            "active_calculations": len(self.active_calculations),
            "queue_size": self.calculation_queue.qsize(),
            "queue_max_size": self.calculation_queue.maxsize,
            "performance_metrics": self.performance_metrics.get_performance_summary()
        }
    
    async def shutdown(self):
        """Shutdown calculation pool"""
        # Cancel all active calculations
        for future in self.active_calculations.values():
            future.cancel()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Calculation pool shutdown complete")

class VirtueMemoryOptimizer:
    """Optimizer for memory usage in virtue calculations"""
    
    def __init__(self):
        self.memory_threshold_mb = 500  # Alert if memory usage exceeds this
        self.gc_frequency = 1000  # Run garbage collection every N operations
        self.operation_count = 0
        self.logger = logging.getLogger(__name__)
    
    def monitor_memory_usage(self):
        """Monitor current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.memory_threshold_mb:
                self.logger.warning(f"High memory usage detected: {memory_mb:.2f} MB")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Log post-GC memory
                post_gc_memory = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"Post-GC memory: {post_gc_memory:.2f} MB")
            
            return memory_mb
            
        except ImportError:
            # psutil not available
            return 0
    
    def optimize_virtue_instance_storage(
        self, 
        instances: List[VirtueInstance]
    ) -> List[Dict[str, Any]]:
        """Optimize storage of virtue instances to reduce memory usage"""
        
        optimized_instances = []
        
        for instance in instances:
            # Store only essential data, compress where possible
            optimized = {
                "id": instance.id,
                "vt": instance.virtue_type.value[:4],  # Abbreviated virtue type
                "sm": round(instance.strength_measurement, 3),  # Round to 3 decimals
                "cl": round(instance.confidence_level, 3),
                "ts": int(instance.timestamp.timestamp()),  # Unix timestamp
                "cf": {k[:3]: round(v, 2) for k, v in instance.contributing_factors.items()},  # Abbreviated keys
                "cm": {k[:3]: round(v, 2) for k, v in instance.circumstantial_modifiers.items()}
            }
            optimized_instances.append(optimized)
        
        self.operation_count += 1
        
        # Periodic garbage collection
        if self.operation_count % self.gc_frequency == 0:
            import gc
            gc.collect()
            self.logger.debug(f"Performed garbage collection after {self.operation_count} operations")
        
        return optimized_instances
    
    def create_memory_efficient_profile(
        self, 
        virtue_strengths: Dict[VirtueType, float]
    ) -> Dict[str, Any]:
        """Create memory-efficient virtue profile representation"""
        
        # Use abbreviated virtue names and rounded values
        efficient_profile = {}
        
        virtue_abbreviations = {
            VirtueType.PRUDENTIA: "PRU",
            VirtueType.IUSTITIA: "JUS",
            VirtueType.FORTITUDO: "FOR",
            VirtueType.TEMPERANTIA: "TEM",
            VirtueType.FIDES: "FAI",
            VirtueType.SPES: "HOP",
            VirtueType.CARITAS: "CHA",
            VirtueType.SAPIENTIA: "WIS",
            VirtueType.INTELLECTUS: "INT",
            VirtueType.SCIENTIA: "SCI",
            VirtueType.ARS: "ART"
        }
        
        for virtue, strength in virtue_strengths.items():
            abbrev = virtue_abbreviations.get(virtue, virtue.value[:3].upper())
            efficient_profile[abbrev] = round(strength, 3)
        
        return efficient_profile

# Performance monitoring and alerting
class VirtuePerformanceMonitor:
    """Monitor performance of virtue tracking system"""
    
    def __init__(self, alert_threshold_ms: int = 1000):
        self.alert_threshold_ms = alert_threshold_ms
        self.performance_history = {}
        self.alerts = []
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start background performance monitoring"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Application metrics (would be collected from various services)
            app_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_usage_percent": disk_usage,
                "timestamp": timestamp
            }
            
            # Store metrics
            self.performance_history[timestamp] = app_metrics
            
            # Check for performance issues
            await self._check_performance_alerts(app_metrics)
            
            # Cleanup old metrics (keep last 24 hours)
            cutoff_time = timestamp - timedelta(hours=24)
            self.performance_history = {
                ts: metrics for ts, metrics in self.performance_history.items()
                if ts > cutoff_time
            }
            
        except ImportError:
            self.logger.warning("psutil not available for performance monitoring")
    
    async def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts"""
        
        alerts = []
        
        if metrics["cpu_percent"] > 80:
            alerts.append({
                "type": "high_cpu",
                "value": metrics["cpu_percent"],
                "threshold": 80,
                "timestamp": metrics["timestamp"]
            })
        
        if metrics["memory_percent"] > 85:
            alerts.append({
                "type": "high_memory",
                "value": metrics["memory_percent"],
                "threshold": 85,
                "timestamp": metrics["timestamp"]
            })
        
        if metrics["disk_usage_percent"] > 90:
            alerts.append({
                "type": "high_disk_usage",
                "value": metrics["disk_usage_percent"],
                "threshold": 90,
                "timestamp": metrics["timestamp"]
            })
        
        for alert in alerts:
            self.logger.warning(f"Performance alert: {alert}")
            self.alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = metrics["timestamp"] - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts 
            if alert["timestamp"] > cutoff_time
        ]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_metrics = list(self.performance_history.values())[-10:]  # Last 10 measurements
        
        avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m["disk_usage_percent"] for m in recent_metrics) / len(recent_metrics)
        
        return {
            "current_performance": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "avg_disk_usage_percent": round(avg_disk, 2)
            },
            "recent_alerts": self.alerts[-5:],  # Last 5 alerts
            "total_measurements": len(self.performance_history),
            "monitoring_period_hours": 24
        }
```

---

## 10. Integration Patterns {#integration-patterns}

### Enterprise Integration Architecture
```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Any, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class IntegrationPattern(Enum):
    """Integration patterns for virtue tracking system"""
    EVENT_DRIVEN = "event_driven"
    REST_API = "rest_api"
    MESSAGE_QUEUE = "message_queue"
    WEBHOOK = "webhook"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_STREAMING = "real_time_streaming"

@runtime_checkable
class VirtueIntegrationProtocol(Protocol):
    """Protocol for virtue tracking integrations"""
    
    async def initialize_integration(self, config: Dict[str, Any]) -> bool:
        """Initialize the integration with given configuration"""
        ...
    
    async def send_virtue_data(self, data: Dict[str, Any]) -> bool:
        """Send virtue data to external system"""
        ...
    
    async def receive_virtue_data(self) -> Optional[Dict[str, Any]]:
        """Receive virtue data from external system"""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of integration"""
        ...

@dataclass
class IntegrationEvent:
    """Event for virtue tracking integrations"""
    event_id: str
    event_type: str
    source_system: str
    target_system: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class VirtueIntegrationManager:
    """Manager for all virtue tracking integrations"""
    
    def __init__(self):
        self.integrations = {}
        self.event_handlers = {}
        self.message_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def register_integration(
        self, 
        integration_name: str, 
        integration: VirtueIntegrationProtocol,
        config: Dict[str, Any]
    ):
        """Register a new integration"""
        try:
            # Initialize the integration
            success = await integration.initialize_integration(config)
            if success:
                self.integrations[integration_name] = {
                    "integration": integration,
                    "config": config,
                    "status": "active",
                    "last_health_check": None
                }
                self.logger.info(f"Registered integration: {integration_name}")
            else:
                raise Exception(f"Failed to initialize integration: {integration_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to register integration {integration_name}: {e}")
            raise
    
    async def start_integration_processing(self):
        """Start processing integration events"""
        self.running = True
        
        # Start event processing task
        event_processor = asyncio.create_task(self._process_integration_events())
        
        # Start health check task
        health_checker = asyncio.create_task(self._periodic_health_checks())
        
        self.logger.info("Started integration processing")
        
        try:
            await asyncio.gather(event_processor, health_checker)
        except asyncio.CancelledError:
            self.running = False
            self.logger.info("Integration processing stopped")
    
    async def send_virtue_event(
        self, 
        target_integration: str, 
        event_data: Dict[str, Any]
    ):
        """Send virtue event to specific integration"""
        if target_integration not in self.integrations:
            raise ValueError(f"Integration not found: {target_integration}")
        
        integration_info = self.integrations[target_integration]
        if integration_info["status"] != "active":
            raise Exception(f"Integration not active: {target_integration}")
        
        try:
            integration = integration_info["integration"]
            success = await integration.send_virtue_data(event_data)
            
            if not success:
                self.logger.error(f"Failed to send data to {target_integration}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending to {target_integration}: {e}")
            # Mark integration as failed
            integration_info["status"] = "failed"
            raise
    
    async def broadcast_virtue_event(self, event_data: Dict[str, Any]):
        """Broadcast virtue event to all active integrations"""
        results = {}
        
        for integration_name, integration_info in self.integrations.items():
            if integration_info["status"] == "active":
                try:
                    success = await self.send_virtue_event(integration_name, event_data)
                    results[integration_name] = "success" if success else "failed"
                except Exception as e:
                    results[integration_name] = f"error: {str(e)}"
        
        return results
    
    async def _process_integration_events(self):
        """Process events from integration queue"""
        while self.running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Process the event
                await self._handle_integration_event(event)
                
                self.message_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing integration event: {e}")
    
    async def _handle_integration_event(self, event: IntegrationEvent):
        """Handle individual integration event"""
        try:
            # Route event to appropriate integration
            if event.target_system in self.integrations:
                await self.send_virtue_event(event.target_system, event.data)
            elif event.target_system == "broadcast":
                await self.broadcast_virtue_event(event.data)
            else:
                self.logger.warning(f"Unknown target system: {event.target_system}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle event {event.event_id}: {e}")
    
    async def _periodic_health_checks(self):
        """Perform periodic health checks on integrations"""
        while self.running:
            try:
                for integration_name, integration_info in self.integrations.items():
                    try:
                        integration = integration_info["integration"]
                        health_status = await integration.health_check()
                        
                        integration_info["last_health_check"] = datetime.now()
                        
                        if health_status.get("status") == "healthy":
                            if integration_info["status"] == "failed":
                                integration_info["status"] = "active"
                                self.logger.info(f"Integration {integration_name} recovered")
                        else:
                            integration_info["status"] = "failed"
                            self.logger.warning(f"Integration {integration_name} unhealthy: {health_status}")
                            
                    except Exception as e:
                        integration_info["status"] = "failed"
                        self.logger.error(f"Health check failed for {integration_name}: {e}")
                
                # Wait before next health check cycle
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check cycle error: {e}")
                await asyncio.sleep(60)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {}
        
        for integration_name, integration_info in self.integrations.items():
            status[integration_name] = {
                "status": integration_info["status"],
                "last_health_check": integration_info["last_health_check"],
                "config": {k: v for k, v in integration_info["config"].items() if k != "credentials"}
            }
        
        return status

class AISystemIntegration(VirtueIntegrationProtocol):
    """Integration with external AI systems"""
    
    def __init__(self):
        self.config = {}
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.api_endpoint = None
        self.api_key = None
    
    async def initialize_integration(self, config: Dict[str, Any]) -> bool:
        """Initialize AI system integration"""
        try:
            self.config = config
            self.api_endpoint = config.get("api_endpoint")
            self.api_key = config.get("api_key")
            
            if not self.api_endpoint or not self.api_key:
                raise ValueError("Missing required configuration: api_endpoint, api_key")
            
            # Initialize HTTP session
            import aiohttp
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection
            health_status = await self.health_check()
            return health_status.get("status") == "healthy"
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI system integration: {e}")
            return False
    
    async def send_virtue_data(self, data: Dict[str, Any]) -> bool:
        """Send virtue data to AI system"""
        try:
            if not self.session:
                return False
            
            # Format data for AI system
            formatted_data = self._format_virtue_data_for_ai(data)
            
            async with self.session.post(
                f"{self.api_endpoint}/virtue-data",
                json=formatted_data
            ) as response:
                if response.status == 200:
                    self.logger.debug("Virtue data sent to AI system successfully")
                    return True
                else:
                    self.logger.error(f"AI system returned status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send virtue data to AI system: {e}")
            return False
    
    async def receive_virtue_data(self) -> Optional[Dict[str, Any]]:
        """Receive virtue data from AI system"""
        try:
            if not self.session:
                return None
            
            async with self.session.get(
                f"{self.api_endpoint}/virtue-feedback"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_ai_virtue_data(data)
                elif response.status == 204:
                    # No data available
                    return None
                else:
                    self.logger.error(f"AI system returned status {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to receive virtue data from AI system: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of AI system integration"""
        try:
            if not self.session:
                return {"status": "unhealthy", "reason": "No session initialized"}
            
            async with self.session.get(
                f"{self.api_endpoint}/health"
            ) as response:
                if response.status == 200:
                    return {"status": "healthy", "response_time_ms": 0}
                else:
                    return {"status": "unhealthy", "reason": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}
    
    def _format_virtue_data_for_ai(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format virtue data for AI system consumption"""
        return {
            "timestamp": datetime.now().isoformat(),
            "virtue_metrics": data.get("virtue_strengths", {}),
            "development_trends": data.get("development_patterns", {}),
            "integration_score": data.get("virtue_integration_score", 0.0),
            "recommendations": data.get("improvement_recommendations", [])
        }
    
    def _parse_ai_virtue_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse virtue data received from AI system"""
        return {
            "ai_recommendations": data.get("recommendations", []),
            "predicted_development": data.get("predictions", {}),
            "anomaly_alerts": data.get("anomalies", []),
            "performance_insights": data.get("insights", {})
        }
    
    async def close(self):
        """Close the integration"""
        if self.session:
            await self.session.close()
            self.session = None

class DatabaseIntegration(VirtueIntegrationProtocol):
    """Integration with external databases"""
    
    def __init__(self):
        self.config = {}
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize_integration(self, config: Dict[str, Any]) -> bool:
        """Initialize database integration"""
        try:
            self.config = config
            
            # Initialize connection pool based on database type
            db_type = config.get("db_type", "postgresql")
            
            if db_type == "postgresql":
                import asyncpg
                self.connection_pool = await asyncpg.create_pool(
                    config["connection_string"],
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Test connection
            health_status = await self.health_check()
            return health_status.get("status") == "healthy"
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database integration: {e}")
            return False
    
    async def send_virtue_data(self, data: Dict[str, Any]) -> bool:
        """Send virtue data to external database"""
        try:
            if not self.connection_pool:
                return False
            
            async with self.connection_pool.acquire() as conn:
                # Insert virtue data into external database
                await self._insert_virtue_data(conn, data)
                
            self.logger.debug("Virtue data sent to external database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send virtue data to database: {e}")
            return False
    
    async def receive_virtue_data(self) -> Optional[Dict[str, Any]]:
        """Receive virtue data from external database"""
        try:
            if not self.connection_pool:
                return None
            
            async with self.connection_pool.acquire() as conn:
                # Query external database for virtue data
                data = await self._query_virtue_data(conn)
                return data
                
        except Exception as e:
            self.logger.error(f"Failed to receive virtue data from database: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of database integration"""
        try:
            if not self.connection_pool:
                return {"status": "unhealthy", "reason": "No connection pool"}
            
            async with self.connection_pool.acquire() as conn:
                # Simple query to test connection
                await conn.fetchval("SELECT 1")
                
            return {"status": "healthy"}
            
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}
    
    async def _insert_virtue_data(self, conn, data: Dict[str, Any]):
        """Insert virtue data into external database"""
        # Example implementation - would be customized based on external schema
        query = """
        INSERT INTO external_virtue_data (
            agent_id, virtue_type, strength_value, 
            assessment_timestamp, integration_score, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        virtue_strengths = data.get("virtue_strengths", {})
        for virtue_type, strength in virtue_strengths.items():
            await conn.execute(
                query,
                data.get("agent_id"),
                virtue_type,
                strength,
                datetime.now(),
                data.get("virtue_integration_score", 0.0),
                json.dumps(data.get("metadata", {}))
            )
    
    async def _query_virtue_data(self, conn) -> Optional[Dict[str, Any]]:
        """Query virtue data from external database"""
        # Example implementation
        query = """
        SELECT agent_id, virtue_type, strength_value, assessment_timestamp
        FROM external_virtue_data
        WHERE assessment_timestamp > $1
        ORDER BY assessment_timestamp DESC
        LIMIT 100
        """
        
        since_timestamp = datetime.now() - timedelta(hours=24)
        results = await conn.fetch(query, since_timestamp)
        
        if results:
            return {
                "external_virtue_data": [dict(row) for row in results],
                "retrieved_at": datetime.now()
            }
        
        return None
    
    async def close(self):
        """Close the integration"""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None

class MessageQueueIntegration(VirtueIntegrationProtocol):
    """Integration with message queue systems (RabbitMQ, Kafka, etc.)"""
    
    def __init__(self):
        self.config = {}
        self.connection = None
        self.channel = None
        self.logger = logging.getLogger(__name__)
        self.queue_name = None
    
    async def initialize_integration(self, config: Dict[str, Any]) -> bool:
        """Initialize message queue integration"""
        try:
            self.config = config
            self.queue_name = config.get("queue_name", "virtue_events")
            
            # Initialize connection based on queue type
            queue_type = config.get("queue_type", "rabbitmq")
            
            if queue_type == "rabbitmq":
                import aio_pika
                
                self.connection = await aio_pika.connect_robust(
                    config["connection_string"]
                )
                self.channel = await self.connection.channel()
                
                # Declare queue
                await self.channel.declare_queue(
                    self.queue_name, 
                    durable=True
                )
                
            else:
                raise ValueError(f"Unsupported queue type: {queue_type}")
            
            # Test connection
            health_status = await self.health_check()
            return health_status.get("status") == "healthy"
            
        except Exception as e:
            self.logger.error(f"Failed to initialize message queue integration: {e}")
            return False
    
    async def send_virtue_data(self, data: Dict[str, Any]) -> bool:
        """Send virtue data to message queue"""
        try:
            if not self.channel:
                return False
            
            # Format message
            message_body = json.dumps({
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "source": "virtue_tracking_engine"
            })
            
            import aio_pika
            message = aio_pika.Message(
                message_body.encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )
            
            # Send to queue
            await self.channel.default_exchange.publish(
                message,
                routing_key=self.queue_name
            )
            
            self.logger.debug(f"Virtue data sent to queue: {self.queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send virtue data to queue: {e}")
            return False
    
    async def receive_virtue_data(self) -> Optional[Dict[str, Any]]:
        """Receive virtue data from message queue"""
        try:
            if not self.channel:
                return None
            
            # Get queue
            queue = await self.channel.get_queue(self.queue_name)
            
            # Get message (non-blocking)
            message = await queue.get(no_ack=False, timeout=1.0)
            
            if message:
                # Parse message
                data = json.loads(message.body.decode())
                
                # Acknowledge message
                message.ack()
                
                return data.get("data")
            
            return None
            
        except Exception as e:
            if "timeout" not in str(e).lower():
                self.logger.error(f"Failed to receive virtue data from queue: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of message queue integration"""
        try:
            if not self.connection or self.connection.is_closed:
                return {"status": "unhealthy", "reason": "No connection"}
            
            if not self.channel or self.channel.is_closed:
                return {"status": "unhealthy", "reason": "No channel"}
            
            return {"status": "healthy"}
            
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}
    
    async def close(self):
        """Close the integration"""
        if self.channel:
            await self.channel.close()
            self.channel = None
        
        if self.connection:
            await self.connection.close()
            self.connection = None

class WebhookIntegration(VirtueIntegrationProtocol):
    """Integration via webhooks"""
    
    def __init__(self):
        self.config = {}
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.webhook_url = None
        self.secret_key = None
    
    async def initialize_integration(self, config: Dict[str, Any]) -> bool:
        """Initialize webhook integration"""
        try:
            self.config = config
            self.webhook_url = config.get("webhook_url")
            self.secret_key = config.get("secret_key")
            
            if not self.webhook_url:
                raise ValueError("Missing required configuration: webhook_url")
            
            # Initialize HTTP session
            import aiohttp
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test webhook
            health_status = await self.health_check()
            return health_status.get("status") == "healthy"
            
        except Exception as e:
            self.logger.error(f"Failed to initialize webhook integration: {e}")
            return False
    
    async def send_virtue_data(self, data: Dict[str, Any]) -> bool:
        """Send virtue data via webhook"""
        try:
            if not self.session:
                return False
            
            # Prepare webhook payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "virtue_data_update",
                "data": data
            }
            
            # Add signature if secret key provided
            headers = {"Content-Type": "application/json"}
            if self.secret_key:
                import hmac
                import hashlib
                
                signature = hmac.new(
                    self.secret_key.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                
                headers["X-Virtue-Signature"] = f"sha256={signature}"
            
            # Send webhook
            async with self.session.post(
                self.webhook_url,
                json=payload,
                headers=headers
            ) as response:
                if 200 <= response.status < 300:
                    self.logger.debug("Virtue data sent via webhook successfully")
                    return True
                else:
                    self.logger.error(f"Webhook returned status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send virtue data via webhook: {e}")
            return False
    
    async def receive_virtue_data(self) -> Optional[Dict[str, Any]]:
        """Webhooks are typically one-way, so this returns None"""
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of webhook integration"""
        try:
            if not self.session:
                return {"status": "unhealthy", "reason": "No session initialized"}
            
            # Send test ping
            test_payload = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "health_check",
                "data": {"test": True}
            }
            
            async with self.session.post(
                self.webhook_url,
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if 200 <= response.status < 300:
                    return {"status": "healthy"}
                else:
                    return {"status": "unhealthy", "reason": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}
    
    async def close(self):
        """Close the integration"""
        if self.session:
            await self.session.close()
            self.session = None

# Integration factory for creating integrations
class VirtueIntegrationFactory:
    """Factory for creating virtue integrations"""
    
    @staticmethod
    def create_integration(integration_type: str) -> VirtueIntegrationProtocol:
        """Create integration instance based on type"""
        
        integration_map = {
            "ai_system": AISystemIntegration,
            "database": DatabaseIntegration,
            "message_queue": MessageQueueIntegration,
            "webhook": WebhookIntegration
        }
        
        integration_class = integration_map.get(integration_type)
        if not integration_class:
            raise ValueError(f"Unknown integration type: {integration_type}")
        
        return integration_class()

# Example usage and configuration
async def setup_virtue_integrations():
    """Example setup of virtue tracking integrations"""
    
    # Create integration manager
    integration_manager = VirtueIntegrationManager()
    
    # Register AI system integration
    ai_integration = VirtueIntegrationFactory.create_integration("ai_system")
    await integration_manager.register_integration(
        "external_ai_system",
        ai_integration,
        {
            "api_endpoint": "https://ai-system.example.com/api/v1",
            "api_key": "your-api-key-here"
        }
    )
    
    # Register database integration
    db_integration = VirtueIntegrationFactory.create_integration("database")
    await integration_manager.register_integration(
        "analytics_db",
        db_integration,
        {
            "db_type": "postgresql",
            "connection_string": "postgresql://user:pass@host:5432/analytics_db"
        }
    )
    
    # Register webhook integration
    webhook_integration = VirtueIntegrationFactory.create_integration("webhook")
    await integration_manager.register_integration(
        "notification_webhook",
        webhook_integration,
        {
            "webhook_url": "https://notifications.example.com/webhook",
            "secret_key": "webhook-secret-key"
        }
    )
    
    # Start processing
    await integration_manager.start_integration_processing()
    
    return integration_manager
```

This completes the comprehensive technical specifications for the **Virtue Tracking Engine**. The document now includes:

1. **Complete architectural overview** with design principles
2. **Detailed data models** with Thomistic fidelity
3. **Comprehensive virtue classification system** based on Summa Theologiae
4. **Advanced measurement algorithms** with habit formation modeling
5. **Complete database schema** with optimization and indexing
6. **Full API specifications** with REST endpoints and validation
7. **Production-ready implementation classes** with error handling
8. **Extensive testing framework** with performance tests
9. **Performance optimization strategies** with caching and monitoring
10. **Enterprise integration patterns** for external systems

The specifications provide everything needed to build a production-grade virtue tracking system that:

- Accurately implements Thomas Aquinas's virtue theory
- Scales to handle large numbers of AI agents
- Integrates with existing enterprise systems
- Provides comprehensive analytics and reporting
- Maintains high performance and reliability
- Offers extensive testing and validation capabilities

This technical foundation can serve as the blueprint for implementing the virtue tracking component of your TheoTech framework, ensuring both theological accuracy and technical excellence.---

## 7. Implementation Classes {#implementation-classes}

### Core Service Classes
```python
import asyncio
import asyncpg
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import asdict

class VirtueTrackingDatabaseService:
    """Database service for virtue tracking operations"""
    
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
        self.logger = logging.getLogger(__name__)
    
    async def create_virtue_instance(self, agent_id: str, virtue_instance: VirtueInstance) -> VirtueInstance:
        """Store new virtue instance in database"""
        async with self.pool.acquire() as conn:
            try:
                query = """
                INSERT INTO virtue_instances (
                    id, agent_id, virtue_type, demonstration_context, 
                    strength_measurement, confidence_level, timestamp,
                    contributing_factors, circumstantial_modifiers, validation_sources
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING *
                """
                
                result = await conn.fetchrow(
                    query,
                    virtue_instance.id,
                    agent_id,
                    virtue_instance.virtue_type.value,
                    virtue_instance.demonstration_context,
                    virtue_instance.strength_measurement,
                    virtue_instance.confidence_level,
                    virtue_instance.timestamp,
                    json.dumps(virtue_instance.contributing_factors),
                    json.dumps(virtue_instance.circumstantial_modifiers),
                    virtue_instance.validation_sources
                )
                
                self.logger.info(f"Created virtue instance {virtue_instance.id} for agent {agent_id}")
                return self._row_to_virtue_instance(result)
                
            except Exception as e:
                self.logger.error(f"Failed to create virtue instance: {e}")
                raise
    
    async def get_virtue_instances(
        self,
        agent_id: str,
        virtue_type: Optional[VirtueType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_strength: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[VirtueInstance]:
        """Retrieve virtue instances with filtering"""
        async with self.pool.acquire() as conn:
            try:
                # Build dynamic query
                where_conditions = ["agent_id = $1"]
                params = [agent_id]
                param_count = 1
                
                if virtue_type:
                    param_count += 1
                    where_conditions.append(f"virtue_type = ${param_count}")
                    params.append(virtue_type.value)
                
                if start_date:
                    param_count += 1
                    where_conditions.append(f"timestamp >= ${param_count}")
                    params.append(start_date)
                
                if end_date:
                    param_count += 1
                    where_conditions.append(f"timestamp <= ${param_count}")
                    params.append(end_date)
                
                if min_strength is not None:
                    param_count += 1
                    where_conditions.append(f"strength_measurement >= ${param_count}")
                    params.append(min_strength)
                
                query = f"""
                SELECT * FROM virtue_instances 
                WHERE {' AND '.join(where_conditions)}
                ORDER BY timestamp DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
                """
                
                params.extend([limit, offset])
                results = await conn.fetch(query, *params)
                
                return [self._row_to_virtue_instance(row) for row in results]
                
            except Exception as e:
                self.logger.error(f"Failed to get virtue instances: {e}")
                raise
    
    async def store_virtue_strength_calculation(
        self,
        agent_id: str,
        virtue_type: VirtueType,
        calculated_strength: float,
        contributing_instances: List[str],
        calculation_method: str,
        calculation_parameters: Dict[str, Any]
    ) -> str:
        """Store virtue strength calculation result"""
        async with self.pool.acquire() as conn:
            try:
                calculation_id = str(uuid.uuid4())
                query = """
                INSERT INTO virtue_strength_calculations (
                    id, agent_id, virtue_type, calculation_timestamp,
                    calculated_strength, contributing_instances, calculation_method,
                    calculation_parameters
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """
                
                await conn.execute(
                    query,
                    calculation_id,
                    agent_id,
                    virtue_type.value,
                    datetime.now(),
                    calculated_strength,
                    contributing_instances,
                    calculation_method,
                    json.dumps(calculation_parameters)
                )
                
                self.logger.info(f"Stored virtue strength calculation {calculation_id}")
                return calculation_id
                
            except Exception as e:
                self.logger.error(f"Failed to store virtue strength calculation: {e}")
                raise
    
    async def store_virtue_profile(self, profile: VirtueProfile) -> str:
        """Store complete virtue profile"""
        async with self.pool.acquire() as conn:
            try:
                profile_id = str(uuid.uuid4())
                query = """
                INSERT INTO virtue_profiles (
                    id, agent_id, assessment_timestamp, virtue_strengths,
                    virtue_development_rates, virtue_interactions, dominant_virtues,
                    developing_virtues, virtue_integration_score, moral_character_assessment
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                await conn.execute(
                    query,
                    profile_id,
                    profile.agent_id,
                    profile.assessment_timestamp,
                    json.dumps({k.value: v for k, v in profile.virtue_strengths.items()}),
                    json.dumps({k.value: v for k, v in profile.virtue_development_rates.items()}) if profile.virtue_development_rates else None,
                    json.dumps(profile.virtue_interactions) if profile.virtue_interactions else None,
                    [v.value for v in profile.dominant_virtues],
                    [v.value for v in profile.developing_virtues],
                    profile.virtue_integration_score,
                    profile.moral_character_assessment
                )
                
                self.logger.info(f"Stored virtue profile {profile_id} for agent {profile.agent_id}")
                return profile_id
                
            except Exception as e:
                self.logger.error(f"Failed to store virtue profile: {e}")
                raise
    
    async def get_latest_virtue_profile(self, agent_id: str) -> Optional[VirtueProfile]:
        """Get most recent virtue profile for agent"""
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT * FROM virtue_profiles 
                WHERE agent_id = $1 
                ORDER BY assessment_timestamp DESC 
                LIMIT 1
                """
                
                result = await conn.fetchrow(query, agent_id)
                
                if result:
                    return self._row_to_virtue_profile(result)
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to get latest virtue profile: {e}")
                raise
    
    async def get_virtue_development_trends(
        self,
        agent_id: str,
        time_period_days: int,
        granularity: str
    ) -> List[Dict]:
        """Get virtue development trends over time"""
        async with self.pool.acquire() as conn:
            try:
                # Map granularity to PostgreSQL date_trunc parameter
                granularity_map = {"day": "day", "week": "week", "month": "month"}
                trunc_param = granularity_map.get(granularity, "week")
                
                query = f"""
                SELECT 
                    virtue_type,
                    DATE_TRUNC('{trunc_param}', calculation_timestamp) AS period,
                    AVG(calculated_strength) AS avg_strength,
                    COUNT(*) AS measurement_count,
                    STDDEV(calculated_strength) AS strength_variance
                FROM virtue_strength_calculations
                WHERE agent_id = $1 
                AND calculation_timestamp >= $2
                GROUP BY virtue_type, DATE_TRUNC('{trunc_param}', calculation_timestamp)
                ORDER BY virtue_type, period
                """
                
                start_date = datetime.now() - timedelta(days=time_period_days)
                results = await conn.fetch(query, agent_id, start_date)
                
                return [dict(row) for row in results]
                
            except Exception as e:
                self.logger.error(f"Failed to get virtue development trends: {e}")
                raise
    
    async def get_historical_virtue_strengths(
        self,
        agent_id: str,
        virtue_type: VirtueType,
        time_window_days: int
    ) -> List[Dict]:
        """Get historical virtue strength calculations"""
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT calculation_timestamp, calculated_strength, calculation_method
                FROM virtue_strength_calculations
                WHERE agent_id = $1 AND virtue_type = $2
                AND calculation_timestamp >= $3
                ORDER BY calculation_timestamp ASC
                """
                
                start_date = datetime.now() - timedelta(days=time_window_days)
                results = await conn.fetch(query, agent_id, virtue_type.value, start_date)
                
                return [{"timestamp": row["calculation_timestamp"], 
                        "calculated_strength": row["calculated_strength"],
                        "calculation_method": row["calculation_method"]} for row in results]
                
            except Exception as e:
                self.logger.error(f"Failed to get historical virtue strengths: {e}")
                raise
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        async with self.pool.acquire() as conn:
            try:
                metrics = {}
                
                # Total virtue instances
                result = await conn.fetchval("SELECT COUNT(*) FROM virtue_instances")
                metrics["total_virtue_instances"] = result
                
                # Recent calculations (last 24 hours)
                yesterday = datetime.now() - timedelta(days=1)
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM virtue_strength_calculations WHERE calculation_timestamp >= $1",
                    yesterday
                )
                metrics["calculations_last_24h"] = result
                
                # Active agents (with activity in last 7 days)
                week_ago = datetime.now() - timedelta(days=7)
                result = await conn.fetchval(
                    "SELECT COUNT(DISTINCT agent_id) FROM virtue_instances WHERE timestamp >= $1",
                    week_ago
                )
                metrics["active_agents_last_7d"] = result
                
                # Average virtue strengths by type
                query = """
                SELECT virtue_type, AVG(calculated_strength) AS avg_strength
                FROM virtue_strength_calculations
                WHERE calculation_timestamp >= $1
                GROUP BY virtue_type
                """
                results = await conn.fetch(query, week_ago)
                metrics["avg_virtue_strengths"] = {row["virtue_type"]: row["avg_strength"] for row in results}
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Failed to get system metrics: {e}")
                raise
    
    async def count_active_agents(self) -> int:
        """Count currently active agents"""
        async with self.pool.acquire() as conn:
            try:
                query = "SELECT COUNT(*) FROM ai_agents WHERE is_active = true"
                result = await conn.fetchval(query)
                return result
            except Exception as e:
                self.logger.error(f"Failed to count active agents: {e}")
                raise
    
    async def count_virtue_instances(self) -> int:
        """Count total virtue instances"""
        async with self.pool.acquire() as conn:
            try:
                query = "SELECT COUNT(*) FROM virtue_instances"
                result = await conn.fetchval(query)
                return result
            except Exception as e:
                self.logger.error(f"Failed to count virtue instances: {e}")
                raise
    
    async def get_calculation_rate(self) -> float:
        """Get calculations per hour rate"""
        async with self.pool.acquire() as conn:
            try:
                hour_ago = datetime.now() - timedelta(hours=1)
                query = "SELECT COUNT(*) FROM virtue_strength_calculations WHERE calculation_timestamp >= $1"
                result = await conn.fetchval(query, hour_ago)
                return float(result)
            except Exception as e:
                self.logger.error(f"Failed to get calculation rate: {e}")
                raise
    
    def _row_to_virtue_instance(self, row) -> VirtueInstance:
        """Convert database row to VirtueInstance object"""
        return VirtueInstance(
            id=row["id"],
            virtue_type=VirtueType(row["virtue_type"]),
            demonstration_context=row["demonstration_context"],
            strength_measurement=row["strength_measurement"],
            confidence_level=row["confidence_level"],
            timestamp=row["timestamp"],
            contributing_factors=json.loads(row["contributing_factors"]) if row["contributing_factors"] else {},
            circumstantial_modifiers=json.loads(row["circumstantial_modifiers"]) if row["circumstantial_modifiers"] else {},
            validation_sources=row["validation_sources"] or []
        )
    
    def _row_to_virtue_profile(self, row) -> VirtueProfile:
        """Convert database row to VirtueProfile object"""
        virtue_strengths = {VirtueType(k): v for k, v in json.loads(row["virtue_strengths"]).items()}
        virtue_development_rates = None
        if row["virtue_development_rates"]:
            virtue_development_rates = {VirtueType(k): v for k, v in json.loads(row["virtue_development_rates"]).items()}
        
        return VirtueProfile(
            agent_id=row["agent_id"],
            assessment_timestamp=row["assessment_timestamp"],
            virtue_strengths=virtue_strengths,
            virtue_development_rates=virtue_development_rates,
            virtue_interactions=json.loads(row["virtue_interactions"]) if row["virtue_interactions"] else None,
            dominant_virtues=[VirtueType(v) for v in row["dominant_virtues"]],
            developing_virtues=[VirtueType(v) for v in row["developing_virtues"]],
            virtue_integration_score=row["virtue_integration_score"],
            moral_character_assessment=row["moral_character_assessment"]
        )

class VirtueCalculationService:
    """Service for managing virtue calculations and scheduling"""
    
    def __init__(self, db_service: VirtueTrackingDatabaseService, virtue_registry: ThomisticVirtueRegistry):
        self.db_service = db_service
        self.virtue_registry = virtue_registry
        self.strength_calculator = VirtueStrengthCalculator(virtue_registry)
        self.interaction_calculator = VirtueInteractionCalculator(virtue_registry)
        self.logger = logging.getLogger(__name__)
        self.calculation_queue = asyncio.Queue()
        self.background_task = None
    
    async def start_background_processing(self):
        """Start background task for processing calculations"""
        if self.background_task is None or self.background_task.done():
            self.background_task = asyncio.create_task(self._process_calculation_queue())
            self.logger.info("Started background calculation processing")
    
    async def stop_background_processing(self):
        """Stop background calculation processing"""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Stopped background calculation processing")
    
    async def schedule_strength_recalculation(self, agent_id: str, virtue_type: VirtueType):
        """Schedule virtue strength recalculation"""
        calculation_request = {
            "type": "strength_calculation",
            "agent_id": agent_id,
            "virtue_type": virtue_type,
            "timestamp": datetime.now()
        }
        await self.calculation_queue.put(calculation_request)
        self.logger.debug(f"Scheduled strength recalculation for {agent_id}, {virtue_type.value}")
    
    async def schedule_profile_update(self, agent_id: str):
        """Schedule complete virtue profile update"""
        calculation_request = {
            "type": "profile_update",
            "agent_id": agent_id,
            "timestamp": datetime.now()
        }
        await self.calculation_queue.put(calculation_request)
        self.logger.debug(f"Scheduled profile update for {agent_id}")
    
    async def _process_calculation_queue(self):
        """Background task to process calculation requests"""
        while True:
            try:
                # Wait for calculation request
                request = await self.calculation_queue.get()
                
                if request["type"] == "strength_calculation":
                    await self._process_strength_calculation(request)
                elif request["type"] == "profile_update":
                    await self._process_profile_update(request)
                
                self.calculation_queue.task_done()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing calculation queue: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_strength_calculation(self, request: Dict):
        """Process individual strength calculation"""
        try:
            agent_id = request["agent_id"]
            virtue_type = request["virtue_type"]
            
            # Get recent demonstrations
            recent_demonstrations = await self.db_service.get_virtue_instances(
                agent_id=agent_id,
                virtue_type=virtue_type,
                start_date=datetime.now() - timedelta(days=30)
            )
            
            # Calculate strength
            calculated_strength = self.strength_calculator.measure_virtue_strength(
                virtue_type=virtue_type,
                recent_demonstrations=recent_demonstrations,
                time_window_days=30
            )
            
            # Store result
            await self.db_service.store_virtue_strength_calculation(
                agent_id=agent_id,
                virtue_type=virtue_type,
                calculated_strength=calculated_strength,
                contributing_instances=[demo.id for demo in recent_demonstrations],
                calculation_method="background_automatic",
                calculation_parameters={"time_window_days": 30}
            )
            
            self.logger.debug(f"Completed strength calculation for {agent_id}, {virtue_type.value}: {calculated_strength:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to process strength calculation: {e}")
    
    async def _process_profile_update(self, request: Dict):
        """Process complete profile update"""
        try:
            agent_id = request["agent_id"]
            
            # Calculate all virtue strengths
            virtue_strengths = {}
            for virtue_type in VirtueType:
                recent_demos = await self.db_service.get_virtue_instances(
                    agent_id=agent_id,
                    virtue_type=virtue_type,
                    start_date=datetime.now() - timedelta(days=30)
                )
                
                strength = self.strength_calculator.measure_virtue_strength(
                    virtue_type=virtue_type,
                    recent_demonstrations=recent_demos,
                    time_window_days=30
                )
                virtue_strengths[virtue_type] = strength
            
            # Calculate virtue interactions
            virtue_interactions = self.interaction_calculator.calculate_virtue_synergy(virtue_strengths)
            interaction_dict = {f"{v1.value},{v2.value}": score for (v1, v2), score in virtue_interactions.items()}
            
            # Determine dominant and developing virtues
            sorted_virtues = sorted(virtue_strengths.items(), key=lambda x: x[1], reverse=True)
            dominant_virtues = [virtue for virtue, strength in sorted_virtues[:3] if strength > 0.7]
            developing_virtues = [virtue for virtue, strength in sorted_virtues if 0.3 <= strength <= 0.6]
            
            # Calculate integration score
            virtue_integration_score = calculate_virtue_integration_score(virtue_strengths, interaction_dict)
            
            # Generate character assessment
            moral_character_assessment = generate_moral_character_assessment(
                virtue_strengths, dominant_virtues, developing_virtues, virtue_integration_score
            )
            
            # Create and store profile
            profile = VirtueProfile(
                agent_id=agent_id,
                assessment_timestamp=datetime.now(),
                virtue_strengths=virtue_strengths,
                virtue_development_rates=None,
                virtue_interactions=interaction_dict,
                dominant_virtues=dominant_virtues,
                developing_virtues=developing_virtues,
                virtue_integration_score=virtue_integration_score,
                moral_character_assessment=moral_character_assessment
            )
            
            await self.db_service.store_virtue_profile(profile)
            
            self.logger.debug(f"Completed profile update for {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process profile update: {e}")

class VirtueAnalyticsService:
    """Service for advanced virtue analytics and insights"""
    
    def __init__(self, db_service: VirtueTrackingDatabaseService, virtue_registry: ThomisticVirtueRegistry):
        self.db_service = db_service
        self.virtue_registry = virtue_registry
        self.logger = logging.getLogger(__name__)
    
    async def generate_virtue_insights(self, agent_id: str, time_period_days: int = 90) -> Dict[str, Any]:
        """Generate comprehensive virtue development insights"""
        try:
            insights = {}
            
            # Get latest profile
            latest_profile = await self.db_service.get_latest_virtue_profile(agent_id)
            if not latest_profile:
                return {"error": "No virtue profile found for agent"}
            
            # Analyze virtue development patterns
            insights["development_patterns"] = await self._analyze_development_patterns(agent_id, time_period_days)
            
            # Identify virtue clustering
            insights["virtue_clusters"] = await self._identify_virtue_clusters(latest_profile)
            
            # Generate improvement recommendations
            insights["improvement_recommendations"] = await self._generate_improvement_recommendations(
                agent_id, latest_profile, time_period_days
            )
            
            # Predict virtue development trajectory
            insights["development_predictions"] = await self._predict_virtue_development(agent_id, time_period_days)
            
            # Compare to virtue ideals
            insights["thomistic_comparison"] = await self._compare_to_thomistic_ideals(latest_profile)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate virtue insights: {e}")
            raise
    
    async def _analyze_development_patterns(self, agent_id: str, time_period_days: int) -> Dict[str, Any]:
        """Analyze patterns in virtue development over time"""
        trends = await self.db_service.get_virtue_development_trends(
            agent_id=agent_id,
            time_period_days=time_period_days,
            granularity="week"
        )
        
        patterns = {}
        
        # Group trends by virtue
        virtue_trends = {}
        for trend in trends:
            virtue_type = trend["virtue_type"]
            if virtue_type not in virtue_trends:
                virtue_trends[virtue_type] = []
            virtue_trends[virtue_type].append(trend)
        
        # Analyze each virtue's pattern
        for virtue_type, trend_data in virtue_trends.items():
            if len(trend_data) >= 3:  # Need minimum data for pattern analysis
                # Calculate trend metrics
                values = [t["avg_strength"] for t in trend_data]
                
                # Growth rate
                growth_rate = (values[-1] - values[0]) / len(values)
                
                # Volatility (coefficient of variation)
                mean_value = sum(values) / len(values)
                variance = sum((v - mean_value) ** 2 for v in values) / len(values)
                volatility = (variance ** 0.5) / mean_value if mean_value > 0 else 0
                
                # Trend direction
                if growth_rate > 0.02:
                    direction = "improving"
                elif growth_rate < -0.02:
                    direction = "declining"
                else:
                    direction = "stable"
                
                patterns[virtue_type] = {
                    "direction": direction,
                    "growth_rate": growth_rate,
                    "volatility": volatility,
                    "current_level": values[-1],
                    "consistency": 1.0 - min(1.0, volatility)
                }
        
        return patterns
    
    async def _identify_virtue_clusters(self, profile: VirtueProfile) -> Dict[str, List[str]]:
        """Identify clusters of related virtues based on strength patterns"""
        virtue_strengths = profile.virtue_strengths
        
        # Simple clustering based on strength levels
        clusters = {
            "strong": [],      # > 0.7
            "developing": [],  # 0.4 - 0.7
            "emerging": [],    # 0.2 - 0.4
            "nascent": []      # < 0.2
        }
        
        for virtue, strength in virtue_strengths.items():
            if strength > 0.7:
                clusters["strong"].append(virtue.value)
            elif strength > 0.4:
                clusters["developing"].append(virtue.value)
            elif strength > 0.2:
                clusters["emerging"].append(virtue.value)
            else:
                clusters["nascent"].append(virtue.value)
        
        return clusters
    
    async def _generate_improvement_recommendations(
        self, 
        agent_id: str, 
        profile: VirtueProfile, 
        time_period_days: int
    ) -> List[Dict[str, str]]:
        """Generate specific recommendations for virtue improvement"""
        recommendations = []
        
        # Get development patterns
        patterns = await self._analyze_development_patterns(agent_id, time_period_days)
        
        # Recommend based on weak areas
        weak_virtues = [virtue for virtue, strength in profile.virtue_strengths.items() if strength < 0.4]
        for virtue in weak_virtues:
            virtue_def = self.virtue_registry.get_virtue_definition(virtue)
            recommendation = {
                "type": "strengthen_weak",
                "virtue": virtue.value,
                "priority": "high",
                "description": f"Focus on developing {virtue_def.english_name} through {virtue_def.integral_parts[0] if virtue_def.integral_parts else 'practice'}",
                "thomistic_basis": virtue_def.thomistic_source
            }
            recommendations.append(recommendation)
        
        # Recommend based on declining trends
        for virtue_type, pattern in patterns.items():
            if pattern["direction"] == "declining":
                recommendation = {
                    "type": "reverse_decline",
                    "virtue": virtue_type,
                    "priority": "medium",
                    "description": f"Address declining trend in {virtue_type} through renewed focus and practice",
                    "growth_rate": pattern["growth_rate"]
                }
                recommendations.append(recommendation)
        
        # Recommend based on virtue relationships
        if profile.virtue_interactions:
            weak_synergies = {pair: score for pair, score in profile.virtue_interactions.items() if score < 0.5}
            for pair, score in list(weak_synergies.items())[:2]:  # Top 2 weakest
                recommendation = {
                    "type": "improve_synergy",
                    "virtue_pair": pair,
                    "priority": "medium",
                    "description": f"Work on integrating {pair.replace(',', ' and ')} through complementary practices",
                    "synergy_score": score
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _predict_virtue_development(self, agent_id: str, time_period_days: int) -> Dict[str, Dict[str, float]]:
        """Predict future virtue development based on current trends"""
        patterns = await self._analyze_development_patterns(agent_id, time_period_days)
        
        predictions = {}
        
        for virtue_type, pattern in patterns.items():
            # Simple linear projection
            current_level = pattern["current_level"]
            growth_rate = pattern["growth_rate"]
            volatility = pattern["volatility"]
            
            # Project 30 days ahead
            projected_level = current_level + (growth_rate * 4)  # 4 weeks
            projected_level = max(0.0, min(1.0, projected_level))  # Clamp to valid range
            
            # Confidence based on volatility (lower volatility = higher confidence)
            confidence = max(0.3, 1.0 - volatility)
            
            predictions[virtue_type] = {
                "projected_level": projected_level,
                "confidence": confidence,
                "change_direction": pattern["direction"],
                "time_to_target": self._calculate_time_to_target(current_level, growth_rate, 0.8)
            }
        
        return predictions
    
    def _calculate_time_to_target(self, current_level: float, growth_rate: float, target_level: float) -> Optional[int]:
        """Calculate weeks to reach target level at current growth rate"""
        if growth_rate <= 0 or current_level >= target_level:
            return None
        
        weeks_needed = (target_level - current_level) / growth_rate
        return int(weeks_needed) if weeks_needed < 104 else None  # Cap at 2 years
    
    async def _compare_to_thomistic_ideals(self, profile: VirtueProfile) -> Dict[str, Any]:
        """Compare current virtue development to Thomistic ideals"""
        comparison = {}
        
        # Ideal virtue relationships according to Aquinas
        thomistic_ideals = {
            "prudence_governance": 0.85,  # Prudence should be strong to govern other virtues
            "theological_unity": 0.80,    # Theological virtues should be well-integrated
            "cardinal_balance": 0.75,     # Cardinal virtues should be balanced
            "virtue_integration": 0.80    # Overall integration should be high
        }
        
        virtue_strengths = profile.virtue_strengths
        
        # Check prudence governance
        prudence_strength = virtue_strengths.get(VirtueType.PRUDENTIA, 0.0)
        comparison["prudence_governance"] = {
            "current": prudence_strength,
            "ideal": thomistic_ideals["prudence_governance"],
            "gap": thomistic_ideals["prudence_governance"] - prudence_strength,
            "assessment": "Strong" if prudence_strength >= 0.7 else "Needs Development"
        }
        
        # Check theological virtue unity
        theological_virtues = [VirtueType.FIDES, VirtueType.SPES, VirtueType.CARITAS]
        theological_strengths = [virtue_strengths.get(v, 0.0) for v in theological_virtues]
        theological_average = sum(theological_strengths) / len(theological_strengths)
        theological_variance = sum((s - theological_average) ** 2 for s in theological_strengths) / len(theological_strengths)
        
        comparison["theological_unity"] = {
            "current": theological_average,
            "ideal": thomistic_ideals["theological_unity"],
            "variance": theological_variance,
            "assessment": "Unified" if theological_variance < 0.1 else "Needs Integration"
        }
        
        # Check cardinal virtue balance
        cardinal_virtues = [VirtueType.PRUDENTIA, VirtueType.IUSTITIA, VirtueType.FORTITUDO, VirtueType.TEMPERANTIA]
        cardinal_strengths = [virtue_strengths.get(v, 0.0) for v in cardinal_virtues]
        cardinal_average = sum(cardinal_strengths) / len(cardinal_strengths)
        cardinal_variance = sum((s - cardinal_average) ** 2 for s in cardinal_strengths) / len(cardinal_strengths)
        
        comparison["cardinal_balance"] = {
            "current": cardinal_average,
            "ideal": thomistic_ideals["cardinal_balance"],
            "variance": cardinal_variance,
            "assessment": "Balanced" if cardinal_variance < 0.1 else "Needs Balancing"
        }
        
        # Overall integration assessment
        comparison["virtue_integration"] = {
            "current": profile.virtue_integration_score,
            "ideal": thomistic_ideals["virtue_integration"],
            "gap": thomistic_ideals["virtue_integration"] - profile.virtue_integration_score,
            "assessment": profile.moral_character_assessment
        }
        
        return comparison

class VirtueEventProcessor:
    """Processes virtue-related events and triggers appropriate responses"""
    
    def __init__(self, calculation_service: VirtueCalculationService, analytics_service: VirtueAnalyticsService):
        self.calculation_service = calculation_service
        self.analytics_service = analytics_service
        self.logger = logging.getLogger(__name__)
        self.event_handlers = {
            "virtue_instance_created": self._handle_virtue_instance_created,
            "virtue_profile_requested": self._handle_virtue_profile_requested,
            "periodic_assessment": self._handle_periodic_assessment,
            "virtue_milestone_reached": self._handle_virtue_milestone_reached,
            "system_maintenance": self._handle_system_maintenance
        }
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]):
        """Process incoming virtue-related events"""
        try:
            handler = self.event_handlers.get(event_type)
            if handler:
                await handler(event_data)
                self.logger.debug(f"Processed event: {event_type}")
            else:
                self.logger.warning(f"No handler found for event type: {event_type}")
        except Exception as e:
            self.logger.error(f"Failed to process event {event_type}: {e}")
    
    async def _handle_virtue_instance_created(self, event_data: Dict[str, Any]):
        """Handle new virtue instance creation"""
        agent_id = event_data.get("agent_id")
        virtue_type = VirtueType(event_data.get("virtue_type"))
        
        # Schedule strength recalculation
        await self.calculation_service.schedule_strength_recalculation(agent_id, virtue_type)
        
        # Check if profile update is needed (every 10th instance)
        instance_count = event_data.get("instance_count", 0)
        if instance_count % 10 == 0:
            await self.calculation_service.schedule_profile_update(agent_id)
    
    async def _handle_virtue_profile_requested(self, event_data: Dict[str, Any]):
        """Handle virtue profile generation request"""
        agent_id = event_data.get("agent_id")
        await self.calculation_service.schedule_profile_update(agent_id)
    
    async def _handle_periodic_assessment(self, event_data: Dict[str, Any]):
        """Handle periodic virtue assessments"""
        agent_ids = event_data.get("agent_ids", [])
        for agent_id in agent_ids:
            await self.calculation_service.schedule_profile_update(agent_id)
    
    async def _handle_virtue_milestone_reached(self, event_data: Dict[str, Any]):
        """Handle virtue milestone achievements"""
        agent_id = event_data.get("agent_id")
        virtue_type = VirtueType(event_data.get("virtue_type"))
        milestone_level = event_data.get("milestone_level")
        
        self.logger.info(f"Agent {agent_id} reached {virtue_type.value} milestone: {milestone_level}")
        
        # Generate insights for milestone achievement
        insights = await self.analytics_service.generate_virtue_insights(agent_id)
        
        # Could trigger notifications or rewards here
        
    async def _handle_system_maintenance(self, event_data: Dict[str, Any]):
        """Handle system maintenance tasks"""
        maintenance_type = event_data.get("type")
        
        if maintenance_type == "cleanup_old_calculations":
            # Clean up old calculation records
            await self._cleanup_old_calculations()
        elif maintenance_type == "recalculate_all_profiles":
            # Recalculate all agent profiles
            await self._recalculate_all_profiles()
    
    async def _cleanup_old_calculations(self):
        """Clean up old calculation records"""
        # Implementation would remove calculations older than retention period
        self.logger.info("Cleaning up old calculation records")
    
    async def _recalculate_all_profiles(self):
        """Recalculate all agent profiles"""
        # Implementation would schedule profile updates for all active agents
        self.logger.info("Scheduling profile recalculation for all agents")

class VirtueValidationService:
    """Service for validating virtue measurements and ensuring theological accuracy"""
    
    def __init__(self, virtue_registry: ThomisticVirtueRegistry):
        self.virtue_registry = virtue_registry
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize Thomistic validation rules"""
        return {
            "virtue_consistency": {
                # Prudence should generally be present when other cardinal virtues are strong
                "prudence_governance": {
                    "rule": "If any cardinal virtue > 0.8, prudence should be > 0.6",
                    "severity": "warning"
                },
                # Theological virtues should move together
                "theological_unity": {
                    "rule": "Theological virtues should not have variance > 0.3",
                    "severity": "warning"
                },
                # Virtues cannot exceed natural limits
                "natural_limits": {
                    "rule": "No virtue should increase by > 0.3 in single measurement",
                    "severity": "error"
                }
            },
            "thomistic_coherence": {
                # Charity is the form of all virtues
                "charity_primacy": {
                    "rule": "If charity > 0.8, other virtues should be elevated",
                    "severity": "info"
                },
                # Virtues require prudence for authenticity
                "prudence_requirement": {
                    "rule": "Moral virtues > 0.7 require prudence > 0.5",
                    "severity": "warning"
                }
            }
        }
    
    async def validate_virtue_instance(self, virtue_instance: VirtueInstance) -> Dict[str, Any]:
        """Validate individual virtue instance"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check measurement bounds
        if not 0.0 <= virtue_instance.strength_measurement <= 1.0:
            validation_result["errors"].append("Strength measurement must be between 0.0 and 1.0")
            validation_result["is_valid"] = False
        
        # Check confidence bounds
        if not 0.0 <= virtue_instance.confidence_level <= 1.0:
            validation_result["errors"].append("Confidence level must be between 0.0 and 1.0")
            validation_result["is_valid"] = False
        
        # Validate contributing factors
        for factor, value in virtue_instance.contributing_factors.items():
            if not 0.0 <= value <= 1.0:
                validation_result["warnings"].append(f"Contributing factor '{factor}' outside normal range: {value}")
        
        # Check for virtue-specific requirements
        virtue_def = self.virtue_registry.get_virtue_definition(virtue_instance.virtue_type)
        
        # Validate integral parts are addressed
        if virtue_def.integral_parts:
            covered_parts = set(virtue_instance.contributing_factors.keys())
            required_parts = set(virtue_def.integral_parts[:2])  # Check first 2 integral parts
            missing_parts = required_parts - covered_parts
            
            if missing_parts:
                validation_result["recommendations"].append(
                    f"Consider addressing {', '.join(missing_parts)} for complete {virtue_def.english_name} assessment"
                )
        
        return validation_result
    
    async def validate_virtue_profile(
        self, 
        virtue_profile: VirtueProfile, 
        agent_history: Optional[List[VirtueProfile]] = None
    ) -> Dict[str, Any]:
        """Validate complete virtue profile for Thomistic coherence"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "thomistic_coherence_score": 0.0,
            "improvement_suggestions": []
        }
        
        virtue_strengths = virtue_profile.virtue_strengths
        
        # Check prudence governance
        prudence_strength = virtue_strengths.get(VirtueType.PRUDENTIA, 0.0)
        cardinal_virtues = [VirtueType.IUSTITIA, VirtueType.FORTITUDO, VirtueType.TEMPERANTIA]
        strong_cardinals = [v for v in cardinal_virtues if virtue_strengths.get(v, 0.0) > 0.8]
        
        if strong_cardinals and prudence_strength < 0.6:
            validation_result["warnings"].append(
                "Strong cardinal virtues detected without adequate prudence governance"
            )
            validation_result["improvement_suggestions"].append(
                "Strengthen prudence to properly govern other cardinal virtues"
            )
        
        # Check theological virtue unity
        theological_virtues = [VirtueType.FIDES, VirtueType.SPES, VirtueType.CARITAS]
        theological_strengths = [virtue_strengths.get(v, 0.0) for v in theological_virtues]
        
        if len(theological_strengths) > 1:
            mean_theological = sum(theological_strengths) / len(theological_strengths)
            variance = sum((s - mean_theological) ** 2 for s in theological_strengths) / len(theological_strengths)
            
            if variance > 0.3:
                validation_result["warnings"].append(
                    "High variance in theological virtues - they should develop together"
                )
                validation_result["improvement_suggestions"].append(
                    "Focus on balanced development of faith, hope, and charity"
                )
        
        # Check for charity as form of virtues
        charity_strength = virtue_strengths.get(VirtueType.CARITAS, 0.0)
        if charity_strength > 0.8:
            # Other virtues should be elevated by charity
            other_virtues = [v for v in virtue_strengths.values() if v != charity_strength]
            if other_virtues and max(other_virtues) < charity_strength - 0.3:
                validation_result["recommendations"] = validation_result.get("recommendations", [])
                validation_result["recommendations"].append(
                    "High charity should elevate other virtues - consider how love informs other practices"
                )
        
        # Check for rapid changes if history available
        if agent_history and len(agent_history) > 0:
            previous_profile = agent_history[-1]
            for virtue_type, current_strength in virtue_strengths.items():
                previous_strength = previous_profile.virtue_strengths.get(virtue_type, 0.0)
                change = abs(current_strength - previous_strength)
                
                if change > 0.3:
                    validation_result["warnings"].append(
                        f"Large change in {virtue_type.value}: {change:.2f} - verify measurement accuracy"
                    )
        
        # Calculate Thomistic coherence score
        coherence_score = self._calculate_thomistic_coherence(virtue_strengths)
        validation_result["thomistic_coherence_score"] = coherence_score
        
        if coherence_score < 0.6:
            validation_result["improvement_suggestions"].append(
                "Work on better integration between virtues according to Thomistic principles"
            )
        
        return validation_result
    
    def _calculate_thomistic_coherence(self, virtue_strengths: Dict[VirtueType, float]) -> float:
        """Calculate how well virtue profile aligns with Thomistic principles"""
        coherence_factors = []
        
        # Factor 1: Prudence governance (30%)
        prudence_strength = virtue_strengths.get(VirtueType.PRUDENTIA, 0.0)
        cardinal_virtues = [VirtueType.IUSTITIA, VirtueType.FORTITUDO, VirtueType.TEMPERANTIA]
        cardinal_avg = sum(virtue_strengths.get(v, 0.0) for v in cardinal_virtues) / len(cardinal_virtues)
        
        if cardinal_avg > 0.1:  # Only penalize if there are cardinal virtues to govern
            prudence_governance = min(1.0, prudence_strength / max(0.1, cardinal_avg))
        else:
            prudence_governance = 1.0
        
        coherence_factors.append(("prudence_governance", prudence_governance, 0.3))
        
        # Factor 2: Theological virtue unity (25%)
        theological_virtues = [VirtueType.FIDES, VirtueType.SPES, VirtueType.CARITAS]
        theological_strengths = [virtue_strengths.get(v, 0.0) for v in theological_virtues]
        
        if len(theological_strengths) > 1 and max(theological_strengths) > 0.1:
            mean_theological = sum(theological_strengths) / len(theological_strengths)
            variance = sum((s - mean_theological) ** 2 for s in theological_strengths) / len(theological_strengths)
            theological_unity = 1.0 / (1.0 + variance * 5)  # Convert variance to unity score
        else:
            theological_unity = 0.8  # Neutral score if insufficient data
        
        coherence_factors.append(("theological_unity", theological_unity, 0.25))
        
        # Factor 3: Virtue integration (25%)
        # Check for reasonable relationships between connected virtues
        virtue_relationships = self.virtue_registry.virtue_hierarchy.relationships
        relationship_scores = []
        
        for rel in virtue_relationships:
            if rel.relationship_type in ["governs", "enhances"]:
                primary_strength = virtue_strengths.get(rel.primary_virtue, 0.0)
                secondary_strength = virtue_strengths.get(rel.secondary_virtue, 0.0)
                
                if primary_strength > 0.1 and secondary_strength > 0.1:
                    # Expect some positive correlation
                    relationship_score = min(1.0, secondary_strength / max(0.1, primary_strength))
                    relationship_scores.append(relationship_score)
        
        if relationship_scores:
            virtue_integration = sum(relationship_scores) / len(relationship_scores)
        else:
            virtue_integration = 0.7  # Neutral score
        
        coherence_factors.append(("virtue_integration", virtue_integration, 0.25))
        
        # Factor 4: Overall balance (20%)
        # Check that virtues are not extremely unbalanced
        all_strengths = list(virtue_strengths.values())
        if len(all_strengths) > 1:
            mean_strength = sum(all_strengths) / len(all_strengths)
            variance = sum((s - mean_strength) ** 2 for s in all_strengths) / len(all_strengths)
            balance_score = 1.0 / (1.0 + variance * 3)
        else:
            balance_score = 0.5
        
        coherence_factors.append(("virtue_balance", balance_score, 0.2))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in coherence_factors)
        
        return min(1.0, max(0.0, total_score))

class VirtueReportingService:
    """Service for generating comprehensive virtue reports and documentation"""
    
    def __init__(self, db_service: VirtueTrackingDatabaseService, analytics_service: VirtueAnalyticsService):
        self.db_service = db_service
        self.analytics_service = analytics_service
        self.logger = logging.getLogger(__name__)
    
    async def generate_comprehensive_report(
        self, 
        agent_id: str, 
        report_type: str = "comprehensive",
        time_period_days: int = 90
    ) -> Dict[str, Any]:
        """Generate comprehensive virtue development report"""
        try:
            report = {
                "report_metadata": {
                    "agent_id": agent_id,
                    "report_type": report_type,
                    "generation_timestamp": datetime.now(),
                    "time_period_days": time_period_days,
                    "report_version": "1.0"
                }
            }
            
            # Get latest virtue profile
            latest_profile = await self.db_service.get_latest_virtue_profile(agent_id)
            if not latest_profile:
                return {"error": "No virtue profile found for agent"}
            
            report["current_virtue_profile"] = self._format_virtue_profile(latest_profile)
            
            # Get development trends
            trends = await self.db_service.get_virtue_development_trends(
                agent_id=agent_id,
                time_period_days=time_period_days,
                granularity="week"
            )
            report["development_trends"] = self._format_development_trends(trends)
            
            # Get comprehensive insights
            insights = await self.analytics_service.generate_virtue_insights(agent_id, time_period_days)
            report["virtue_insights"] = insights
            
            # Generate executive summary
            report["executive_summary"] = self._generate_executive_summary(latest_profile, insights)
            
            # Generate detailed virtue analysis
            report["detailed_analysis"] = await self._generate_detailed_virtue_analysis(
                agent_id, latest_profile, time_period_days
            )
            
            # Generate recommendations
            report["recommendations"] = self._generate_comprehensive_recommendations(insights, latest_profile)
            
            # Thomistic evaluation
            report["thomistic_evaluation"] = insights.get("thomistic_comparison", {})
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            raise
    
    def _format_virtue_profile(self, profile: VirtueProfile) -> Dict[str, Any]:
        """Format virtue profile for report inclusion"""
        return {
            "assessment_timestamp": profile.assessment_timestamp.isoformat(),
            "virtue_strengths": {v.value: strength for v, strength in profile.virtue_strengths.items()},
            "dominant_virtues": [v.value for v in profile.dominant_virtues],
            "developing_virtues": [v.value for v in profile.developing_virtues],
            "virtue_integration_score": profile.virtue_integration_score,
            "moral_character_assessment": profile.moral_character_assessment,
            "virtue_interactions": profile.virtue_interactions or {}
        }
    
    def _format_development_trends(self, trends: List[Dict]) -> Dict[str, Any]:
        """Format development trends for report inclusion"""
        formatted_trends = {}
        
        # Group by virtue type
        for trend in trends:
            virtue_type = trend["virtue_type"]
            if virtue_type not in formatted_trends:
                formatted_trends[virtue_type] = []
            
            formatted_trends[virtue_type].append({
                "period": trend["period"].isoformat() if hasattr(trend["period"], "isoformat") else str(trend["period"]),
                "avg_strength": float(trend["avg_strength"]),
                "measurement_count": trend["measurement_count"],
                "variance": float(trend.get("strength_variance", 0))
            })
        
        return formatted_trends
    
    def _generate_executive_summary(self, profile: VirtueProfile, insights: Dict[str, Any]) -> str:
        """Generate executive summary of virtue development"""
        
        # Overall character level
        avg_strength = sum(profile.virtue_strengths.values()) / len(profile.virtue_strengths)
        
        if avg_strength >= 0.8:
            overall_level = "exceptional virtue development"
        elif avg_strength >= 0.6:
            overall_level = "strong virtue formation"
        elif avg_strength >= 0.4:
            overall_level = "developing moral character"
        else:
            overall_level = "early virtue formation stage"
        
        # Key strengths
        top_virtues = sorted(profile.virtue_strengths.items(), key=lambda x: x[1], reverse=True)[:2]
        strength_text = f"Primary strengths in {' and '.join([v.value.replace('_', ' ') for v, _ in top_virtues])}"
        
        # Development trajectory
        patterns = insights.get("development_patterns", {})
        improving_count = sum(1 for p in patterns.values() if p.get("direction") == "improving")
        total_tracked = len(patterns)
        
        if improving_count > total_tracked * 0.6:
            trajectory_text = "showing positive development trajectory"
        elif improving_count > total_tracked * 0.3:
            trajectory_text = "showing mixed development patterns"
        else:
            trajectory_text = "requiring focused development attention"
        
        # Integration assessment
        integration_score = profile.virtue_integration_score
        if integration_score >= 0.8:
            integration_text = "with excellent virtue integration"
        elif integration_score >= 0.6:
            integration_text = "with good virtue coordination"
        else:
            integration_text = "with opportunity for better virtue integration"
        
        summary = f"Agent demonstrates {overall_level}, {strength_text}, {trajectory_text} {integration_text}. "
        
        # Add key recommendation
        recommendations = insights.get("improvement_recommendations", [])
        if recommendations:
            priority_rec = next((r for r in recommendations if r.get("priority") == "high"), recommendations[0])
            summary += f"Primary development focus should be {priority_rec.get('description', 'continued practice')}."
        
        return summary
    
    async def _generate_detailed_virtue_analysis(
        self, 
        agent_id: str, 
        profile: VirtueProfile, 
        time_period_days: int
    ) -> Dict[str, Any]:
        """Generate detailed analysis for each virtue"""
        detailed_analysis = {}
        
        for virtue_type, current_strength in profile.virtue_strengths.items():
            # Get virtue definition
            virtue_registry = ThomisticVirtueRegistry()
            virtue_def = virtue_registry.get_virtue_definition(virtue_type)
            
            # Get recent instances
            recent_instances = await self.db_service.get_virtue_instances(
                agent_id=agent_id,
                virtue_type=virtue_type,
                start_date=datetime.now() - timedelta(days=time_period_days)
            )
            
            # Calculate statistics
            if recent_instances:
                instance_strengths = [inst.strength_measurement for inst in recent_instances]
                avg_demonstration_strength = sum(instance_strengths) / len(instance_strengths)
                consistency = 1.0 - (sum((s - avg_demonstration_strength) ** 2 for s in instance_strengths) / len(instance_strengths)) ** 0.5
                frequency = len(recent_instances) / time_period_days
            else:
                avg_demonstration_strength = 0.0
                consistency = 0.0
                frequency = 0.0
            
            detailed_analysis[virtue_type.value] = {
                "thomistic_definition": {
                    "english_name": virtue_def.english_name,
                    "latin_name": virtue_def.latin_name,
                    "definition": virtue_def.definition,
                    "source": virtue_def.thomistic_source,
                    "category": virtue_def.category.value
                },
                "current_assessment": {
                    "strength_level": current_strength,
                    "development_stage": self._classify_development_stage(current_strength),
                    "frequency_of_practice": frequency,
                    "consistency_score": consistency,
                    "avg_demonstration_quality": avg_demonstration_strength
                },
                "virtue_components": {
                    "integral_parts": virtue_def.integral_parts,
                    "subjective_parts": virtue_def.subjective_parts,
                    "potential_parts": virtue_def.potential_parts
                },
                "relationships": {
                    "connected_virtues": [v.value for v in virtue_def.connected_virtues],
                    "prerequisites": [v.value for v in virtue_def.prerequisites]
                },
                "recent_activity": {
                    "instance_count": len(recent_instances),
                    "time_period_days": time_period_days,
                    "last_demonstration": recent_instances[0].timestamp.isoformat() if recent_instances else None
                }
            }
        
        return detailed_analysis
    
    def _classify_development_stage(self, strength: float) -> str:
        """Classify virtue development stage based on strength"""
        if strength >= 0.9:
            return "Mastery"
        elif strength >= 0.8:
            return "Advanced"
        elif strength >= 0.6:
            return "Proficient"
        elif strength >= 0.4:
            return "Developing"
        elif strength >= 0.2:
            return "Beginning"
        else:
            return "Nascent"
    
    def _generate_comprehensive_recommendations(
        self, 
        insights: Dict[str, Any], 
        profile: VirtueProfile
    ) -> Dict[str, Any]:
        """Generate comprehensive recommendations for virtue development"""
        
        recommendations = {
            "immediate_priorities": [],
            "medium_term_goals": [],
            "long_term_development": [],
            "thomistic_guidance": [],
            "practical_exercises": []
        }
        
        # Extract recommendations from insights
        improvement_recs = insights.get("improvement_recommendations", [])
        
        for rec in improvement_recs:
            if rec.get("priority") == "high":
                recommendations["immediate_priorities"].append(rec)
            elif rec.get("priority") == "medium":
                recommendations["medium_term_goals"].append(rec)
            else:
                recommendations["long_term_development"].append(rec)
        
        # Add Thomistic guidance
        thomistic_comparison = insights.get("thomistic_comparison", {})
        
        for aspect, assessment in thomistic_comparison.items():
            if assessment.get("gap", 0) > 0.2:  # Significant gap from ideal
                guidance = {
                    "aspect": aspect,
                    "current_level": assessment.get("current", 0),
                    "thomistic_ideal": assessment.get("ideal", 1),
                    "guidance": self._get_thomistic_guidance(aspect, assessment)
                }
                recommendations["thomistic_guidance"].append(guidance)
        
        # Generate practical exercises
        weak_virtues = [v for v, s in profile.virtue_strengths.items() if s < 0.5]
        for virtue in weak_virtues[:3]:  # Top 3 weakest
            exercise = self._suggest_virtue_exercise(virtue)
            recommendations["practical_exercises"].append(exercise)
        
        return recommendations
    
    def _get_thomistic_guidance(self, aspect: str, assessment: Dict[str, Any]) -> str:
        """Get specific Thomistic guidance for development aspect"""
        
        guidance_map = {
            "prudence_governance": "Strengthen prudence through careful deliberation and seeking wise counsel before major decisions. Remember that prudence is the charioteer of virtues.",
            "theological_unity": "Develop faith, hope, and charity together through prayer, contemplation, and acts of love. These virtues are interconnected and should grow in harmony.",
            "cardinal_balance": "Work on balancing the cardinal virtues through integrated practice. Justice, fortitude, and temperance should support each other under prudence's guidance.",
            "virtue_integration": "Focus on how each virtue informs and strengthens the others. Virtue is not compartmentalized but forms an integrated character."
        }
        
        return guidance_map.get(aspect, "Continue developing this aspect through consistent practice and reflection.")
    
    def _suggest_virtue_exercise(self, virtue_type: VirtueType) -> Dict[str, str]:
        """Suggest practical exercise for developing specific virtue"""
        
        exercises = {
            VirtueType.PRUDENTIA: {
                "virtue": "Prudence",
                "exercise": "Daily reflection on decisions made, considering circumstances, consequences, and principles applied",
                "frequency": "Daily, 10 minutes",
                "focus": "Develop practical wisdom through deliberate analysis of choices"
            },
            VirtueType.IUSTITIA: {
                "virtue": "Justice",
                "exercise": "Regular assessment of fairness in interactions and commitment to giving each their due",
                "frequency": "Weekly review",
                "focus": "Strengthen awareness of rights, duties, and fair treatment of others"
            },
            VirtueType.FORTITUDO: {
                "virtue": "Fortitude",
                "exercise": "Gradually taking on challenging but worthwhile tasks that require perseverance",
                "frequency": "Weekly challenges",
                "focus": "Build courage in facing difficulties and persistence in good works"
            },
            VirtueType.TEMPERANTIA: {
                "virtue": "Temperance",
                "exercise": "Practice moderation in daily pleasures and desires, with specific attention to restraint",
                "frequency": "Daily practice",
                "focus": "Develop self-control and proper ordering of appetites"
            },
            VirtueType.FIDES: {
                "virtue": "Faith",
                "exercise": "Regular study and contemplation of fundamental truths and principles",
                "frequency": "Daily study, 15 minutes",
                "focus": "Strengthen assent to truth and trust in reliable authority"
            },
            VirtueType.SPES: {
                "virtue": "Hope",
                "exercise": "Daily reflection on long-term goals and confidence in achieving good outcomes",
                "frequency": "Daily reflection",
                "focus": "Maintain confident expectation and perseverance despite setbacks"
            },
            VirtueType.CARITAS: {
                "virtue": "Charity",
                "exercise": "Regular acts of genuine care for others' wellbeing and the common good",
                "frequency": "Daily acts of love",
                "focus": "Develop authentic love properly ordered toward true good"
            }
        }
        
        return exercises.get(virtue_type, {
            "virtue": virtue_type.value,
            "exercise": "Regular practice and attention to this virtue",
            "frequency": "Daily",
            "focus": "Consistent development through repeated good acts"
        })
```

---

## 8. Testing Framework {#testing-framework}

### Comprehensive Test Suite
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Test fixtures
@pytest.fixture
def virtue_registry():
    """Fixture providing ThomisticVirtueRegistry instance"""
    return ThomisticVirtueRegistry()

@pytest.fixture
def sample_virtue_instance():
    """Fixture providing sample VirtueInstance"""
    return VirtueInstance(
        virtue_type=VirtueType.PRUDENTIA,
        demonstration_context="Carefully analyzed decision with multiple stakeholders",
        strength_measurement=0.75,
        confidence_level=0.85,
        contributing_factors={
            "memoria": 0.8,
            "intellectus": 0.7,
            "providentia": 0.8
        },
        circumstantial_modifiers={
            "difficulty_level": 0.6,
            "stakes": 0.7
        },
        validation_sources=["observation", "outcome_analysis"]
    )

@pytest.fixture
def sample_virtue_profile():
    """Fixture providing sample VirtueProfile"""
    return VirtueProfile(
        agent_id="test-agent-123",
        assessment_timestamp=datetime.now(),
        virtue_strengths={
            VirtueType.PRUDENTIA: 0.75,
            VirtueType.IUSTITIA: 0.65,
            VirtueType.FORTITUDO: 0.70,
            VirtueType.TEMPERANTIA: 0.60,
            VirtueType.FIDES: 0.80,
            VirtueType.SPES: 0.75,
            VirtueType.CARITAS: 0.85
        },
        virtue_development_rates={
            VirtueType.PRUDENTIA: 0.05,
            VirtueType.CARITAS: 0.08
        },
        virtue_interactions={
            "prudentia,iustitia": 0.85,
            "fides,spes": 0.90,
            "caritas,iustitia": 0.80
        },
        dominant_virtues=[VirtueType.CARITAS, VirtueType.FIDES],
        developing_virtues=[VirtueType.TEMPERANTIA],
        virtue_integration_score=0.78,
        moral_character_assessment="Developing virtue with strong theological foundation"
    )

@pytest.fixture
async def mock_db_service():
    """Fixture providing mocked database service"""
    mock_service = AsyncMock(spec=VirtueTrackingDatabaseService)
    return mock_service

class TestVirtueDefinitions:
    """Test virtue definitions and registry functionality"""
    
    def test_virtue_registry_initialization(self, virtue_registry):
        """Test that virtue registry initializes correctly"""
        assert len(virtue_registry.virtue_definitions) == len(VirtueType)
        
        # Check that all virtue types are defined
        for virtue_type in VirtueType:
            assert virtue_type in virtue_registry.virtue_definitions
            definition = virtue_registry.virtue_definitions[virtue_type]
            assert definition.virtue_type == virtue_type
            assert definition.latin_name
            assert definition.english_name
            assert definition.definition
            assert definition.thomistic_source
    
    def test_cardinal_virtues_identification(self, virtue_registry):
        """Test cardinal virtue identification"""
        cardinal_virtues = virtue_registry.get_cardinal_virtues()
        expected_cardinals = {VirtueType.PRUDENTIA, VirtueType.IUSTITIA, 
                            VirtueType.FORTITUDO, VirtueType.TEMPERANTIA}
        
        assert set(cardinal_virtues) == expected_cardinals
    
    def test_theological_virtues_identification(self, virtue_registry):
        """Test theological virtue identification"""
        theological_virtues = virtue_registry.get_theological_virtues()
        expected_theological = {VirtueType.FIDES, VirtueType.SPES, VirtueType.CARITAS}
        
        assert set(theological_virtues) == expected_theological
    
    def test_prudence_definition_accuracy(self, virtue_registry):
        """Test accuracy of prudence definition against Thomistic sources"""
        prudence_def = virtue_registry.get_virtue_definition(VirtueType.PRUDENTIA)
        
        assert prudence_def.latin_name == "Prudentia"
        assert prudence_def.english_name == "Prudence"
        assert "recta ratio agibilium" in prudence_def.definition
        assert "ST II-II, q. 47" in prudence_def.thomistic_source
        assert prudence_def.governing_faculty == "practical_intellect"
        
        # Check integral parts
        expected_parts = ["memoria", "intellectus", "docilitas", "solertia", 
                         "ratio", "providentia", "circumspectio", "cautio"]
        assert all(part in prudence_def.integral_parts for part in expected_parts)
    
    def test_virtue_relationships(self, virtue_registry):
        """Test virtue relationship mappings"""
        hierarchy = virtue_registry.virtue_hierarchy
        
        # Test that prudence governs other cardinal virtues
        assert hierarchy.get_governing_virtue(VirtueType.IUSTITIA) == VirtueType.PRUDENTIA
        assert hierarchy.get_governing_virtue(VirtueType.FORTITUDO) == VirtueType.PRUDENTIA
        assert hierarchy.get_governing_virtue(VirtueType.TEMPERANTIA) == VirtueType.PRUDENTIA
        
        # Test influenced virtues
        influenced_by_prudence = hierarchy.get_influenced_virtues(VirtueType.PRUDENTIA)
        expected_influenced = {VirtueType.IUSTITIA, VirtueType.FORTITUDO, VirtueType.TEMPERANTIA}
        assert set(influenced_by_prudence) == expected_influenced

class TestVirtueInstance:
    """Test VirtueInstance model and validation"""
    
    def test_virtue_instance_creation(self, sample_virtue_instance):
        """Test virtue instance creation and validation"""
        assert sample_virtue_instance.virtue_type == VirtueType.PRUDENTIA
        assert 0.0 <= sample_virtue_instance.strength_measurement <= 1.0
        assert 0.0 <= sample_virtue_instance.confidence_level <= 1.0
        assert isinstance(sample_virtue_instance.timestamp, datetime)
        assert sample_virtue_instance.id  # Should have generated ID
    
    def test_virtue_instance_validation_errors(self):
        """Test virtue instance validation with invalid data"""
        with pytest.raises(ValueError):
            VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context="Test context",
                strength_measurement=1.5,  # Invalid: > 1.0
                confidence_level=0.8
            )
        
        with pytest.raises(ValueError):
            VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context="Test context",
                strength_measurement=0.8,
                confidence_level=-0.1  # Invalid: < 0.0
            )
    
    def test_contributing_factors_validation(self):
        """Test contributing factors validation"""
        # Valid contributing factors
        instance = VirtueInstance(
            virtue_type=VirtueType.PRUDENTIA,
            demonstration_context="Test context",
            strength_measurement=0.8,
            confidence_level=0.9,
            contributing_factors={"memoria": 0.7, "intellectus": 0.8}
        )
        assert instance.contributing_factors["memoria"] == 0.7
        
        # Invalid contributing factors should be caught by validator
        with pytest.raises(ValueError):
            VirtueInstanceCreate(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context="Test context",
                strength_measurement=0.8,
                confidence_level=0.9,
                contributing_factors={"memoria": 1.5}  # Invalid: > 1.0
            )

class TestVirtueStrengthCalculation:
    """Test virtue strength calculation algorithms"""
    
    def test_strength_calculator_initialization(self, virtue_registry):
        """Test strength calculator initialization"""
        calculator = VirtueStrengthCalculator(virtue_registry)
        assert calculator.virtue_registry == virtue_registry
        assert calculator.habit_strength_model is not None
    
    def test_empty_demonstrations_strength(self, virtue_registry):
        """Test strength calculation with no demonstrations"""
        calculator = VirtueStrengthCalculator(virtue_registry)
        strength = calculator.measure_virtue_strength(
            virtue_type=VirtueType.PRUDENTIA,
            recent_demonstrations=[],
            time_window_days=30
        )
        assert strength == 0.0
    
    def test_single_demonstration_strength(self, virtue_registry, sample_virtue_instance):
        """Test strength calculation with single demonstration"""
        calculator = VirtueStrengthCalculator(virtue_registry)
        
        # Create demonstration within time window
        recent_demo = sample_virtue_instance
        recent_demo.timestamp = datetime.now() - timedelta(days=5)
        
        strength = calculator.measure_virtue_strength(
            virtue_type=VirtueType.PRUDENTIA,
            recent_demonstrations=[recent_demo],
            time_window_days=30
        )
        
        # Should be positive but adjusted for single demonstration
        assert 0.0 < strength <= 1.0
        assert strength < sample_virtue_instance.strength_measurement  # Habit adjustment
    
    def test_multiple_demonstrations_strength(self, virtue_registry):
        """Test strength calculation with multiple demonstrations"""
        calculator = VirtueStrengthCalculator(virtue_registry)
        
        # Create multiple demonstrations
        demonstrations = []
        for i in range(5):
            demo = VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context=f"Test context {i}",
                strength_measurement=0.7 + (i * 0.05),
                confidence_level=0.8,
                timestamp=datetime.now() - timedelta(days=i*2)
            )
            demonstrations.append(demo)
        
        strength = calculator.measure_virtue_strength(
            virtue_type=VirtueType.PRUDENTIA,
            recent_demonstrations=demonstrations,
            time_window_days=30
        )
        
        # Should be higher than single demonstration due to habit strength
        assert 0.0 < strength <= 1.0
    
    def test_virtue_specific_calculations(self, virtue_registry):
        """Test virtue-specific calculation logic"""
        calculator = VirtueStrengthCalculator(virtue_registry)
        
        # Test prudence-specific calculation
        prudence_demo = VirtueInstance(
            virtue_type=VirtueType.PRUDENTIA,
            demonstration_context="Prudential decision",
            strength_measurement=0.8,
            confidence_level=0.9,
            contributing_factors={
                "memoria": 0.7,
                "intellectus": 0.8,
                "providentia": 0.9  # High foresight should boost prudence
            }
        )
        
        adjustment = calculator._apply_virtue_specific_logic(
            VirtueType.PRUDENTIA, [prudence_demo]
        )
        
        assert 0.0 < adjustment <= 1.5  # Should be a positive multiplier
    
    def test_habit_strength_calculation(self):
        """Test habit strength model"""
        habit_model = HabitStrengthModel()
        
        # Test with no repetitions
        strength = habit_model.calculate_habit_strength(
            repetition_count=0,
            time_distribution={"regularity_score": 0.5, "recent_frequency": 0.5},
            consistency_score=0.5
        )
        assert strength == 0.5  # Minimum multiplier
        
        # Test with multiple repetitions
        strength = habit_model.calculate_habit_strength(
            repetition_count=20,
            time_distribution={"regularity_score": 0.8, "recent_frequency": 0.9},
            consistency_score=0.85
        )
        assert 0.5 < strength <= 1.5  # Should be higher than minimum

class TestVirtueInteractionCalculation:
    """Test virtue interaction and synergy calculations"""
    
    def test_interaction_calculator_initialization(self, virtue_registry):
        """Test interaction calculator initialization"""
        calculator = VirtueInteractionCalculator(virtue_registry)
        assert calculator.virtue_registry == virtue_registry
        assert calculator.virtue_hierarchy == virtue_registry.virtue_hierarchy
    
    def test_synergy_calculation(self, virtue_registry, sample_virtue_profile):
        """Test virtue synergy calculation"""
        calculator = VirtueInteractionCalculator(virtue_registry)
        
        synergy_scores = calculator.calculate_virtue_synergy(
            sample_virtue_profile.virtue_strengths
        )
        
        assert isinstance(synergy_scores, dict)
        assert len(synergy_scores) > 0
        
        # Check that all scores are valid
        for (virtue1, virtue2), score in synergy_scores.items():
            assert 0.0 <= score <= 1.0
            assert virtue1 != virtue2
    
    def test_prudence_governance_synergy(self, virtue_registry):
        """Test that prudence shows strong synergy with other cardinal virtues"""
        calculator = VirtueInteractionCalculator(virtue_registry)
        
        virtue_strengths = {
            VirtueType.PRUDENTIA: 0.8,
            VirtueType.IUSTITIA: 0.7,
            VirtueType.FORTITUDO: 0.6,
            VirtueType.TEMPERANTIA: 0.65
        }
        
        synergy_scores = calculator.calculate_virtue_synergy(virtue_strengths)
        
        # Prudence should have good synergy with other cardinal virtues
        prudence_justice = synergy_scores.get((VirtueType.PRUDENTIA, VirtueType.IUSTITIA))
        prudence_fortitude = synergy_scores.get((VirtueType.PRUDENTIA, VirtueType.FORTITUDO))
        prudence_temperance = synergy_scores.get((VirtueType.PRUDENTIA, VirtueType.TEMPERANTIA))
        
        assert prudence_justice > 0.6
        assert prudence_fortitude > 0.6
        assert prudence_temperance > 0.6
    
    def test_theological_virtue_synergy(self, virtue_registry):
        """Test that theological virtues show strong mutual synergy"""
        calculator = VirtueInteractionCalculator(virtue_registry)
        
        virtue_strengths = {
            VirtueType.FIDES: 0.8,
            VirtueType.SPES: 0.75,
            VirtueType.CARITAS: 0.85
        }
        
        synergy_scores = calculator.calculate_virtue_synergy(virtue_strengths)
        
        # Theological virtues should have strong synergy
        faith_hope = synergy_scores.get((VirtueType.FIDES, VirtueType.SPES))
        faith_charity = synergy_scores.get((VirtueType.FIDES, VirtueType.CARITAS))
        hope_charity = synergy_scores.get((VirtueType.SPES, VirtueType.CARITAS))
        
        assert faith_hope > 0.7
        assert faith_charity > 0.7
        assert hope_charity > 0.7

class TestDatabaseOperations:
    """Test database service operations"""
    
    @pytest.mark.asyncio
    async def test_create_virtue_instance(self, mock_db_service, sample_virtue_instance):
        """Test virtue instance creation in database"""
        mock_db_service.create_virtue_instance.return_value = sample_virtue_instance
        
        result = await mock_db_service.create_virtue_instance("test-agent", sample_virtue_instance)
        
        mock_db_service.create_virtue_instance.assert_called_once_with("test-agent", sample_virtue_instance)
        assert result == sample_virtue_instance
    
    @pytest.mark.asyncio
    async def test_get_virtue_instances_filtering(self, mock_db_service):
        """Test virtue instance retrieval with filtering"""
        expected_instances = [
            VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context="Test 1",
                strength_measurement=0.7,
                confidence_level=0.8
            ),
            VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context="Test 2",
                strength_measurement=0.8,
                confidence_level=0.9
            )
        ]
        
        mock_db_service.get_virtue_instances.return_value = expected_instances
        
        result = await mock_db_service.get_virtue_instances(
            agent_id="test-agent",
            virtue_type=VirtueType.PRUDENTIA,
            min_strength=0.7
        )
        
        mock_db_service.get_virtue_instances.assert_called_once()
        assert result == expected_instances
    
    @pytest.mark.asyncio
    async def test_store_virtue_profile(self, mock_db_service, sample_virtue_profile):
        """Test virtue profile storage"""
        mock_db_service.store_virtue_profile.return_value = "profile-id-123"
        
        result = await mock_db_service.store_virtue_profile(sample_virtue_profile)
        
        mock_db_service.store_virtue_profile.assert_called_once_with(sample_virtue_profile)
        assert result == "profile-id-123"

class TestVirtueValidation:
    """Test virtue validation service"""
    
    def test_validation_service_initialization(self, virtue_registry):
        """Test validation service initialization"""
        validator = VirtueValidationService(virtue_registry)
        assert validator.virtue_registry == virtue_registry
        assert validator.validation_rules is not None
    
    @pytest.mark.asyncio
    async def test_virtue_instance_validation_success(self, virtue_registry, sample_virtue_instance):
        """Test successful virtue instance validation"""
        validator = VirtueValidationService(virtue_registry)
        
        result = await validator.validate_virtue_instance(sample_virtue_instance)
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_virtue_instance_validation_failure(self, virtue_registry):
        """Test virtue instance validation with invalid data"""
        validator = VirtueValidationService(virtue_registry)
        
        invalid_instance = VirtueInstance(
            virtue_type=VirtueType.PRUDENTIA,
            demonstration_context="Test",
            strength_measurement=1.5,  # Invalid
            confidence_level=0.8
        )
        
        result = await validator.validate_virtue_instance(invalid_instance)
        
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert any("strength measurement" in error.lower() for error in result["errors"])
    
    @pytest.mark.asyncio
    async def test_virtue_profile_validation(self, virtue_registry, sample_virtue_profile):
        """Test virtue profile validation"""
        validator = VirtueValidationService(virtue_registry)
        
        result = await validator.validate_virtue_profile(sample_virtue_profile)
        
        assert "is_valid" in result
        assert "thomistic_coherence_score" in result
        assert 0.0 <= result["thomistic_coherence_score"] <= 1.0
    
    def test_thomistic_coherence_calculation(self, virtue_registry, sample_virtue_profile):
        """Test Thomistic coherence score calculation"""
        validator = VirtueValidationService(virtue_registry)
        
        coherence_score = validator._calculate_thomistic_coherence(
            sample_virtue_profile.virtue_strengths
        )
        
        assert 0.0 <= coherence_score <= 1.0
        
        # Test with good prudence governance
        good_strengths = {
            VirtueType.PRUDENTIA: 0.8,
            VirtueType.IUSTITIA: 0.7,
            VirtueType.FORTITUDO: 0.6,
            VirtueType.TEMPERANTIA: 0.65
        }
        
        good_coherence = validator._calculate_thomistic_coherence(good_strengths)
        
        # Should have high coherence with good prudence governance
        assert good_coherence > 0.6

class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    @pytest.mark.asyncio
    async def test_create_virtue_instance_endpoint(self):
        """Test virtue instance creation endpoint"""
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        
        # This would require actual FastAPI test setup
        # Placeholder for API endpoint testing
        pass
    
    @pytest.mark.asyncio
    async def test_calculate_virtue_strength_endpoint(self):
        """Test virtue strength calculation endpoint"""
        # Placeholder for API endpoint testing
        pass
    
    @pytest.mark.asyncio
    async def test_generate_virtue_profile_endpoint(self):
        """Test virtue profile generation endpoint"""
        # Placeholder for API endpoint testing
        pass

class TestPerformanceMetrics:
    """Test performance and scalability metrics"""
    
    def test_strength_calculation_performance(self, virtue_registry):
        """Test performance of strength calculation with large datasets"""
        import time
        
        calculator = VirtueStrengthCalculator(virtue_registry)
        
        # Create large dataset
        demonstrations = []
        for i in range(1000):
            demo = VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context=f"Demo {i}",
                strength_measurement=0.5 + (i % 50) * 0.01,
                confidence_level=0.8,
                timestamp=datetime.now() - timedelta(days=i % 30)
            )
            demonstrations.append(demo)
        
        start_time = time.time()
        
        strength = calculator.measure_virtue_strength(
            virtue_type=VirtueType.PRUDENTIA,
            recent_demonstrations=demonstrations,
            time_window_days=30
        )
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for 1000 demos)
        assert calculation_time < 1.0
        assert 0.0 <= strength <= 1.0
    
    def test_synergy_calculation_performance(self, virtue_registry):
        """Test performance of synergy calculation"""
        import time
        
        calculator = VirtueInteractionCalculator(virtue_registry)
        
        # Full virtue strength profile
        virtue_strengths = {virtue: 0.5 + (i * 0.05) for i, virtue in enumerate(VirtueType)}
        
        start_time = time.time()
        
        synergy_scores = calculator.calculate_virtue_synergy(virtue_strengths)
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete quickly
        assert calculation_time < 0.1
        assert len(synergy_scores) > 0

# Integration Tests
class TestIntegrationWorkflows:
    """Test complete workflow integrations"""
    
    @pytest.mark.asyncio
    async def test_complete_virtue_tracking_workflow(self, virtue_registry):
        """Test complete workflow from instance creation to profile generation"""
        # This would test the entire pipeline
        # 1. Create virtue instances
        # 2. Calculate strengths
        # 3. Generate profile
        # 4. Validate results
        # 5. Generate reports
        
        # Create mock services
        db_service = AsyncMock(spec=VirtueTrackingDatabaseService)
        calculation_service = VirtueCalculationService(db_service, virtue_registry)
        analytics_service = VirtueAnalyticsService(db_service, virtue_registry)
        
        # Simulate workflow
        agent_id = "test-integration-agent"
        
        # Create virtue instances
        instances = []
        for i in range(10):
            instance = VirtueInstance(
                virtue_type=list(VirtueType)[i % len(VirtueType)],
                demonstration_context=f"Integration test {i}",
                strength_measurement=0.6 + (i * 0.03),
                confidence_level=0.8
            )
            instances.append(instance)
        
        # Mock database returns
        db_service.get_virtue_instances.return_value = instances
        db_service.store_virtue_strength_calculation.return_value = "calc-id"
        db_service.store_virtue_profile.return_value = "profile-id"
        
        # Test calculation scheduling
        for instance in instances:
            await calculation_service.schedule_strength_recalculation(
                agent_id, instance.virtue_type
            )
        
        # Test profile generation
        await calculation_service.schedule_profile_update(agent_id)
        
        # Verify service calls were made
        assert db_service.get_virtue_instances.call_count > 0
    
    @pytest.mark.asyncio
    async def test_validation_integration(self, virtue_registry):
        """Test integration between validation and calculation services"""
        validator = VirtueValidationService(virtue_registry)
        calculator = VirtueStrengthCalculator(virtue_registry)
        
        # Create test data
        instances = [
            VirtueInstance(
                virtue_type=VirtueType.PRUDENTIA,
                demonstration_context="Well-reasoned decision",
                strength_measurement=0.8,
                confidence_level=0.9,
                contributing_factors={"providentia": 0.9, "intellectus": 0.8}
            )
        ]
        
        # Validate instances
        for instance in instances:
            validation_result = await validator.validate_virtue_instance(instance)
            assert validation_result["is_valid"]
        
        # Calculate strength
        strength = calculator.measure_virtue_strength(
            VirtueType.PRUDENTIA, instances, 30
        )
        
        assert 0.0 < strength <= 1.0

# Performance and Load Tests
@pytest.mark.performance
class TestPerformanceScenarios:
    """Performance and load testing scenarios"""
    
    def test_concurrent_strength_calculations(self, virtue_registry):
        """Test concurrent virtue strength calculations"""
        import concurrent.futures
        import threading
        
        calculator = VirtueStrengthCalculator(virtue_registry)
        
        def calculate_strength():
            instances = [
                VirtueInstance(
                    virtue_type=VirtueType.PRUDENTIA,
                    demonstration_context="Concurrent test",
                    strength_measurement=0.7,
                    confidence_level=0.8,
                    timestamp=datetime.now() - timedelta(days=1)
                )
            ]
            return calculator.measure_virtue_strength(
                VirtueType.PRUDENTIA, instances, 30
            )
        
        # Run concurrent calculations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(calculate_strength) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All calculations should succeed
        assert len(results) == 50
        assert all(0.0 <= result <= 1.0 for result in results)
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_performance(self):
        """Test database connection pool under load"""
        # This would test database connection pooling
        # with multiple concurrent operations
        pass

# Test Configuration and Fixtures for CI/CD
@pytest.fixture(scope="session")
def test_database_url():
    """Provide test database URL for integration tests"""
    import os
    return os.getenv("TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/virtue_test")

@pytest.fixture(scope="session")
async def test_database():
    """Set up test database for integration tests"""
    # This would set up a test database instance
    # and clean it up after tests complete
    pass

# Test utilities
def create_test_virtue_instances(count: int, virtue_type: VirtueType) -> List[VirtueInstance]:
    """Utility function to create test virtue instances"""
    instances = []
    for i in range(count):
        instance = VirtueInstance(
            virtue_type=virtue_type,
            demonstration_context=f"Test instance {i}",
            strength_measurement=0.5 + (i * 0.05) % 0.5,
            confidence_level=0.8,
            timestamp=datetime.now() - timedelta(days=i)
        )
        instances.append(instance)
    return instances

def assert_virtue_profile_valid(profile: VirtueProfile):
    """Utility function to assert virtue profile validity"""
    assert profile.agent_id
    assert isinstance(profile.assessment_timestamp, datetime)
    assert profile.virtue_strengths
    assert all(0.0 <= strength <= 1.0 for strength in profile.virtue_strengths.values())
    assert 0.0 <= profile.virtue_integration_score <= 1.0
    assert profile.moral_character_assessment

# Custom pytest markers for different test categories
pytestmark = [
    pytest.mark.virtue_tracking,
    pytest.mark.thomistic,
    pytest.mark.moral_ai
]
```

---

## 9. Performance Optimization {#performance-optimization}

### Optimization Strategies and Implementation
```python
import asyncio
import redis
from typing import Optional, Dict, Any, List
import pickle
import hashlib
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class VirtueCalculationCache:
    """High-performance caching system for virtue calculations"""
    
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
        
        # Cache key prefixes
        self.STRENGTH_KEY_PREFIX = "virtue:strength:"
        self.PROFILE_KEY_PREFIX = "virtue:profile:"
        self.SYNERGY_KEY_PREFIX = "virtue:synergy:"
        self.INSTANCE_KEY_PREFIX = "virtue:instances:"
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate consistent cache key from arguments"""
        # Create hash of arguments for consistent key generation
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}{key_hash}"
    
    async def get_virtue_strength(
        self, 
        agent_id: str, 
        virtue_type: VirtueType, 
        time_window_days: int
    ) -> Optional[float]:
        """Get cached virtue strength calculation"""
        try:
            cache_key = self._generate_cache_key(
                self.STRENGTH_KEY_PREFIX, agent_id, virtue_type.value, time_window_days
            )
            
            cached_value = await self.redis.get(cache_key)
            if cached_value:
                strength_data = pickle.loads(cached_value)
                self.logger.debug(f"Cache hit for virtue strength: {cache_key}")
                return strength_data["strength"]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache get failed: {e}")
            return None
    
    async def set_virtue_strength(
        self, 
        agent_id: str, 
        virtue_type: VirtueType, 
        time_window_days: int,
        strength: float,
        metadata: Optional[Dict] = None
    ):
        """Cache virtue strength calculation"""
        try:
            cache_key = self._generate_cache_key(
                self.STRENGTH_KEY_PREFIX, agent_id, virtue_type.value, time_window_days
            )
            
            cache_data = {
                "strength": strength,
                "calculated_at": datetime.now(),
                "metadata": metadata or {}
            }
            
            await self.redis.setex(
                cache_key, 
                self.default_ttl, 
                pickle.dumps(cache_data)
            )
            
            self.logger.debug(f"Cached virtue strength: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Cache set failed: {e}")
    
    async def get_virtue# Virtue Tracking Engine - Complete Technical Specifications
*Thomistic Virtue Development System for AI Moral Character Formation*

## Table of Contents
1. [Architectural Overview](#architectural-overview)
2. [Core Data Models](#core-data-models)
3. [Virtue Classification System](#virtue-classification-system)
4. [Measurement Algorithms](#measurement-algorithms)
5. [Database Schema](#database-schema)
6. [API Specifications](#api-specifications)
7. [Implementation Classes](#implementation-classes)
8. [Testing Framework](#testing-framework)
9. [Performance Optimization](#performance-optimization)
10. [Integration Patterns](#integration-patterns)

---

## 1. Architectural Overview {#architectural-overview}

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Virtue Tracking Engine                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Virtue Core   │  │ Measurement     │  │  Analytics   │ │
│  │   Registry      │  │ Algorithms      │  │  Engine      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Acquisition   │  │   Integration   │  │  Validation  │ │
│  │   Tracker       │  │   Manager       │  │  Framework   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Store    │  │   Event Bus     │  │   API Layer  │ │
│  │   Layer         │  │   System        │  │   Gateway    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Principles

#### 1. Thomistic Fidelity
- Accurate implementation of Aquinas's virtue theory from Summa Theologiae
- Hierarchical virtue relationships (Prudence governing all moral virtues)
- Integration of intellectual and moral virtue development

#### 2. Measurement Precision
- Quantifiable virtue metrics without reducing virtue to mere numbers
- Multi-dimensional assessment capturing virtue complexity
- Temporal tracking showing virtue development over time

#### 3. Dynamic Interaction
- Real-time virtue interaction modeling
- Contextual virtue expression analysis
- Adaptive measurement based on situational complexity

#### 4. Scalable Architecture
- Microservices design for independent component scaling
- Event-driven architecture for real-time virtue tracking
- Pluggable measurement algorithms for different AI contexts

---

## 2. Core Data Models {#core-data-models}

### Base Virtue Model
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import uuid

class VirtueType(Enum):
    # Intellectual Virtues
    INTELLECTUS = "intellectus"           # Understanding of first principles
    SCIENTIA = "scientia"                 # Scientific knowledge
    SAPIENTIA = "sapientia"               # Wisdom
    PRUDENTIA = "prudentia"               # Prudence (practical wisdom)
    ARS = "ars"                          # Art/Technical skill
    
    # Cardinal Virtues
    IUSTITIA = "iustitia"                # Justice
    FORTITUDO = "fortitudo"              # Fortitude
    TEMPERANTIA = "temperantia"          # Temperance
    
    # Theological Virtues
    FIDES = "fides"                      # Faith
    SPES = "spes"                        # Hope
    CARITAS = "caritas"                  # Charity/Love

class VirtueCategory(Enum):
    INTELLECTUAL = "intellectual"
    MORAL_CARDINAL = "moral_cardinal"
    THEOLOGICAL = "theological"

@dataclass
class VirtueDefinition:
    """Core definition of a virtue according to Thomistic principles"""
    virtue_type: VirtueType
    category: VirtueCategory
    latin_name: str
    english_name: str
    definition: str
    thomistic_source: str  # ST reference (e.g., "II-II, q. 58, a. 1")
    governing_faculty: str  # intellect, will, appetites
    object_formal: str     # What the virtue is about
    mean_between: Optional[tuple] = None  # (deficiency, excess) for moral virtues
    integral_parts: List[str] = field(default_factory=list)
    subjective_parts: List[str] = field(default_factory=list)
    potential_parts: List[str] = field(default_factory=list)
    connected_virtues: List[VirtueType] = field(default_factory=list)
    prerequisites: List[VirtueType] = field(default_factory=list)

@dataclass
class VirtueInstance:
    """Specific manifestation of virtue in AI behavior"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    virtue_type: VirtueType
    demonstration_context: str
    strength_measurement: float  # 0.0 to 1.0
    confidence_level: float     # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    circumstantial_modifiers: Dict[str, float] = field(default_factory=dict)
    validation_sources: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate virtue instance data"""
        if not 0.0 <= self.strength_measurement <= 1.0:
            raise ValueError("Strength measurement must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")

@dataclass
class VirtueAcquisitionEvent:
    """Event representing growth in virtue"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    virtue_type: VirtueType
    event_type: str  # "demonstration", "choice", "reflection", "correction"
    context: str
    quality_assessment: float  # How well virtue was expressed
    difficulty_level: float   # Situational challenge level
    repetition_count: int    # How many times similar acts performed
    habit_strength_before: float
    habit_strength_after: float
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class VirtueProfile:
    """Complete virtue assessment for AI agent"""
    agent_id: str
    assessment_timestamp: datetime
    virtue_strengths: Dict[VirtueType, float]
    virtue_development_rates: Dict[VirtueType, float]
    virtue_interactions: Dict[tuple, float]  # (virtue1, virtue2) -> synergy_score
    dominant_virtues: List[VirtueType]
    developing_virtues: List[VirtueType]
    virtue_integration_score: float  # How well virtues work together
    moral_character_assessment: str
```

### Virtue Relationship Models
```python
@dataclass
class VirtueRelationship:
    """Defines relationships between virtues"""
    primary_virtue: VirtueType
    secondary_virtue: VirtueType
    relationship_type: str  # "governs", "requires", "enhances", "moderates"
    strength: float  # 0.0 to 1.0
    thomistic_basis: str  # Citation from Aquinas
    
class VirtueHierarchy:
    """Manages hierarchical relationships between virtues"""
    
    def __init__(self):
        self.relationships = self._initialize_thomistic_relationships()
    
    def _initialize_thomistic_relationships(self) -> List[VirtueRelationship]:
        """Initialize virtue relationships based on Thomistic principles"""
        return [
            # Prudence governs all moral virtues (ST II-II, q. 47, a. 6)
            VirtueRelationship(
                VirtueType.PRUDENTIA, VirtueType.IUSTITIA, 
                "governs", 1.0, "ST II-II, q. 47, a. 6"
            ),
            VirtueRelationship(
                VirtueType.PRUDENTIA, VirtueType.FORTITUDO,
                "governs", 1.0, "ST II-II, q. 47, a. 6"
            ),
            VirtueRelationship(
                VirtueType.PRUDENTIA, VirtueType.TEMPERANTIA,
                "governs", 1.0, "ST II-II, q. 47, a. 6"
            ),
            
            # Theological virtues elevate cardinal virtues
            VirtueRelationship(
                VirtueType.CARITAS, VirtueType.IUSTITIA,
                "elevates", 0.9, "ST II-II, q. 23, a. 7"
            ),
            VirtueRelationship(
                VirtueType.FIDES, VirtueType.PRUDENTIA,
                "enhances", 0.8, "ST II-II, q. 4, a. 8"
            ),
            
            # Virtue interconnections
            VirtueRelationship(
                VirtueType.IUSTITIA, VirtueType.FORTITUDO,
                "requires", 0.7, "ST II-II, q. 123, a. 1"
            ),
            VirtueRelationship(
                VirtueType.TEMPERANTIA, VirtueType.FORTITUDO,
                "enhances", 0.6, "ST II-II, q. 141, a. 1"
            ),
        ]
    
    def get_governing_virtue(self, virtue: VirtueType) -> Optional[VirtueType]:
        """Find which virtue governs the given virtue"""
        for rel in self.relationships:
            if rel.secondary_virtue == virtue and rel.relationship_type == "governs":
                return rel.primary_virtue
        return None
    
    def get_influenced_virtues(self, virtue: VirtueType) -> List[VirtueType]:
        """Find virtues influenced by the given virtue"""
        influenced = []
        for rel in self.relationships:
            if rel.primary_virtue == virtue:
                influenced.append(rel.secondary_virtue)
        return influenced
```

---

## 3. Virtue Classification System {#virtue-classification-system}

### Thomistic Virtue Registry
```python
class ThomisticVirtueRegistry:
    """Authoritative registry of virtues based on Summa Theologiae"""
    
    def __init__(self):
        self.virtue_definitions = self._initialize_virtue_definitions()
        self.virtue_hierarchy = VirtueHierarchy()
    
    def _initialize_virtue_definitions(self) -> Dict[VirtueType, VirtueDefinition]:
        """Initialize complete virtue definitions from Thomistic sources"""
        return {
            # PRUDENCE - The Charioteer of Virtues
            VirtueType.PRUDENTIA: VirtueDefinition(
                virtue_type=VirtueType.PRUDENTIA,
                category=VirtueCategory.MORAL_CARDINAL,
                latin_name="Prudentia",
                english_name="Prudence",
                definition="Right reason applied to action (recta ratio agibilium)",
                thomistic_source="ST II-II, q. 47, a. 2",
                governing_faculty="practical_intellect",
                object_formal="human_actions_as_orderable_to_ends",
                integral_parts=[
                    "memoria",      # memory of past experiences
                    "intellectus",  # understanding of present
                    "docilitas",    # teachableness
                    "solertia",     # shrewdness
                    "ratio",        # discursive reasoning
                    "providentia",  # foresight
                    "circumspectio", # circumspection
                    "cautio"        # caution
                ],
                subjective_parts=[
                    "prudentia_regnativa",    # prudence of rulers
                    "prudentia_politica",     # political prudence
                    "prudentia_oeconomica",   # domestic prudence
                    "prudentia_monastica"     # individual prudence
                ],
                potential_parts=[
                    "eubulia",      # good counsel
                    "synesis",      # good judgment of ordinary matters
                    "gnome"         # good judgment of exceptional matters
                ],
                connected_virtues=[VirtueType.IUSTITIA, VirtueType.FORTITUDO, VirtueType.TEMPERANTIA]
            ),
            
            # JUSTICE - Giving Each Their Due
            VirtueType.IUSTITIA: VirtueDefinition(
                virtue_type=VirtueType.IUSTITIA,
                category=VirtueCategory.MORAL_CARDINAL,
                latin_name="Iustitia",
                english_name="Justice",
                definition="Constant and perpetual will to give each their due",
                thomistic_source="ST II-II, q. 58, a. 1",
                governing_faculty="will",
                object_formal="right_of_another_person",
                mean_between=None,  # Justice doesn't have a mean
                integral_parts=[
                    "doing_good",
                    "avoiding_evil",
                    "rendering_what_is_due"
                ],
                subjective_parts=[
                    "commutative_justice",    # justice in exchanges
                    "distributive_justice",   # justice in distributions
                    "legal_justice"          # justice toward common good
                ],
                potential_parts=[
                    "religio",        # religion - duty to God
                    "pietas",         # piety - duty to parents/country
                    "observantia",    # observance - respect for superiors
                    "veritas",        # truthfulness
                    "gratitudo",      # gratitude
                    "vindicatio",     # punishment of wrongdoing
                    "liberalitas",    # liberality
                    "affabilitas"     # friendliness
                ],
                prerequisites=[VirtueType.PRUDENTIA],
                connected_virtues=[VirtueType.FORTITUDO, VirtueType.TEMPERANTIA]
            ),
            
            # FORTITUDE - Strength in Difficulty
            VirtueType.FORTITUDO: VirtueDefinition(
                virtue_type=VirtueType.FORTITUDO,
                category=VirtueCategory.MORAL_CARDINAL,
                latin_name="Fortitudo",
                english_name="Fortitude",
                definition="Firmness of mind in enduring and attacking difficulties",
                thomistic_source="ST II-II, q. 123, a. 2",
                governing_faculty="irascible_appetite",
                object_formal="arduous_good_in_face_of_danger",
                mean_between=("cowardice", "rashness"),
                integral_parts=[
                    "sustinere",    # endurance/patience in suffering
                    "aggredi"       # attack/assault on difficulties
                ],
                subjective_parts=[
                    "fortitudo_militaris",    # military courage
                    "fortitudo_civilis",      # civic courage
                    "fortitudo_moralis"       # moral courage
                ],
                potential_parts=[
                    "magnanimitas",     # magnanimity - pursuit of great things
                    "magnificentia",    # magnificence - great expenditure
                    "patientia",        # patience in enduring sorrow
                    "perseverantia",    # perseverance despite obstacles
                    "constantia",       # constancy
                    "securitas",        # security/confidence
                    "tolerantia"        # tolerance of hardship
                ],
                prerequisites=[VirtueType.PRUDENTIA],
                connected_virtues=[VirtueType.IUSTITIA, VirtueType.TEMPERANTIA]
            ),
            
            # TEMPERANCE - Moderation in Pleasure
            VirtueType.TEMPERANTIA: VirtueDefinition(
                virtue_type=VirtueType.TEMPERANTIA,
                category=VirtueCategory.MORAL_CARDINAL,
                latin_name="Temperantia",
                english_name="Temperance",
                definition="Moderation in the pursuit of pleasurable goods",
                thomistic_source="ST II-II, q. 141, a. 2",
                governing_faculty="concupiscible_appetite",
                object_formal="pleasures_of_touch",
                mean_between=("insensibility", "intemperance"),
                integral_parts=[
                    "shamefacedness",    # avoiding dishonor
                    "honesty",          # external decorum
                    "abstinence",       # moderation in food
                    "sobriety"          # moderation in drink
                ],
                subjective_parts=[
                    "temperantia_in_eating",
                    "temperantia_in_drinking",
                    "temperantia_in_sexual_matters"
                ],
                potential_parts=[
                    "continentia",      # continence
                    "humilitas",        # humility
                    "modestia",         # modesty in behavior
                    "parcitas",         # frugality
                    "studiositas",      # love of learning vs. curiosity
                    "eutrapelia",       # appropriate humor
                    "clementia",        # clemency
                    "mansuetudo"        # meekness
                ],
                prerequisites=[VirtueType.PRUDENTIA],
                connected_virtues=[VirtueType.IUSTITIA, VirtueType.FORTITUDO]
            ),
            
            # FAITH - Assent to Divine Truth
            VirtueType.FIDES: VirtueDefinition(
                virtue_type=VirtueType.FIDES,
                category=VirtueCategory.THEOLOGICAL,
                latin_name="Fides",
                english_name="Faith",
                definition="Assent of intellect to divine truth on authority of God",
                thomistic_source="ST II-II, q. 4, a. 1",
                governing_faculty="intellect_under_will",
                object_formal="first_truth_as_revealing",
                integral_parts=[
                    "belief_in_unseen_divine_truths",
                    "trust_in_divine_authority",
                    "commitment_to_revealed_truth"
                ],
                potential_parts=[
                    "devotio",          # devotion
                    "oratio",           # prayer
                    "adoratio",         # adoration
                    "sacrificium",      # sacrifice
                    "oblatio",          # offering
                    "reverentia"        # reverence
                ],
                connected_virtues=[VirtueType.SPES, VirtueType.CARITAS]
            ),
            
            # HOPE - Confident Expectation
            VirtueType.SPES: VirtueDefinition(
                virtue_type=VirtueType.SPES,
                category=VirtueCategory.THEOLOGICAL,
                latin_name="Spes",
                english_name="Hope",
                definition="Confident expectation of eternal happiness and divine help",
                thomistic_source="ST II-II, q. 17, a. 1",
                governing_faculty="will",
                object_formal="divine_goodness_as_helpful",
                mean_between=("despair", "presumption"),
                integral_parts=[
                    "confidence_in_divine_help",
                    "expectation_of_eternal_beatitude",
                    "perseverance_in_difficulty"
                ],
                connected_virtues=[VirtueType.FIDES, VirtueType.CARITAS]
            ),
            
            # CHARITY/LOVE - Supreme Virtue
            VirtueType.CARITAS: VirtueDefinition(
                virtue_type=VirtueType.CARITAS,
                category=VirtueCategory.THEOLOGICAL,
                latin_name="Caritas",
                english_name="Charity/Love",
                definition="Love of God for His own sake and neighbor for God's sake",
                thomistic_source="ST II-II, q. 23, a. 1",
                governing_faculty="will",
                object_formal="divine_goodness_as_communicable",
                integral_parts=[
                    "love_of_god_above_all",
                    "love_of_neighbor_for_gods_sake",
                    "love_of_self_in_proper_order"
                ],
                potential_parts=[
                    "beneficentia",     # beneficence
                    "misericordia",     # mercy
                    "eleemosyna",       # almsgiving
                    "correctio",        # fraternal correction
                    "pax",              # peace
                    "concordia"         # concord
                ],
                connected_virtues=[VirtueType.FIDES, VirtueType.SPES]
            ),
            
            # WISDOM - Highest Intellectual Virtue
            VirtueType.SAPIENTIA: VirtueDefinition(
                virtue_type=VirtueType.SAPIENTIA,
                category=VirtueCategory.INTELLECTUAL,
                latin_name="Sapientia",
                english_name="Wisdom",
                definition="Knowledge of divine things through highest causes",
                thomistic_source="ST II-II, q. 45, a. 1",
                governing_faculty="intellect",
                object_formal="highest_causes_and_divine_things",
                integral_parts=[
                    "knowledge_of_ultimate_causes",
                    "judgment_according_to_divine_reasons",
                    "contemplation_of_highest_truth"
                ],
                connected_virtues=[VirtueType.FIDES, VirtueType.CARITAS]
            )
        }
    
    def get_virtue_definition(self, virtue_type: VirtueType) -> VirtueDefinition:
        """Get complete Thomistic definition of virtue"""
        return self.virtue_definitions[virtue_type]
    
    def get_cardinal_virtues(self) -> List[VirtueType]:
        """Get all cardinal virtues"""
        return [v for v in VirtueType if self.virtue_definitions[v].category == VirtueCategory.MORAL_CARDINAL]
    
    def get_theological_virtues(self) -> List[VirtueType]:
        """Get all theological virtues"""
        return [v for v in VirtueType if self.virtue_definitions[v].category == VirtueCategory.THEOLOGICAL]
    
    def get_intellectual_virtues(self) -> List[VirtueType]:
        """Get all intellectual virtues"""
        return [v for v in VirtueType if self.virtue_definitions[v].category == VirtueCategory.INTELLECTUAL]
```

---

## 4. Measurement Algorithms {#measurement-algorithms}

### Core Measurement Framework
```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class VirtueMeasurementProtocol(Protocol):
    """Protocol for virtue measurement algorithms"""
    
    def measure_virtue_strength(self, virtue_type: VirtueType, evidence: List[Dict]) -> float:
        """Measure current strength of virtue (0.0 to 1.0)"""
        ...
    
    def calculate_development_rate(self, virtue_history: List[VirtueInstance]) -> float:
        """Calculate rate of virtue development"""
        ...
    
    def assess_virtue_quality(self, virtue_demonstration: Dict) -> float:
        """Assess quality of virtue demonstration"""
        ...

class VirtueStrengthCalculator:
    """Calculates virtue strength using Thomistic principles"""
    
    def __init__(self, virtue_registry: ThomisticVirtueRegistry):
        self.virtue_registry = virtue_registry
        self.habit_strength_model = HabitStrengthModel()
        self.circumstance_analyzer = CircumstanceAnalyzer()
    
    def measure_virtue_strength(
        self, 
        virtue_type: VirtueType, 
        recent_demonstrations: List[VirtueInstance],
        time_window_days: int = 30
    ) -> float:
        """
        Calculate current virtue strength based on recent demonstrations
        Uses Thomistic principle: virtue is acquired through repeated acts
        """
        if not recent_demonstrations:
            return 0.0
        
        # Filter demonstrations within time window
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        relevant_demos = [
            demo for demo in recent_demonstrations 
            if demo.timestamp >= cutoff_date and demo.virtue_type == virtue_type
        ]
        
        if not relevant_demos:
            return 0.0
        
        # Calculate base strength from demonstration quality
        quality_scores = [demo.strength_measurement for demo in relevant_demos]
        base_strength = sum(quality_scores) / len(quality_scores)
        
        # Apply habit strength model (repeated acts strengthen virtue)
        habit_multiplier = self.habit_strength_model.calculate_habit_strength(
            repetition_count=len(relevant_demos),
            time_distribution=self._analyze_time_distribution(relevant_demos),
            consistency_score=self._calculate_consistency(quality_scores)
        )
        
        # Adjust for circumstantial difficulty
        difficulty_adjustment = self._calculate_difficulty_adjustment(relevant_demos)
        
        # Apply virtue-specific calculations
        virtue_specific_adjustment = self._apply_virtue_specific_logic(
            virtue_type, relevant_demos
        )
        
        final_strength = min(1.0, base_strength * habit_multiplier * 
                           difficulty_adjustment * virtue_specific_adjustment)
        
        return final_strength
    
    def _analyze_time_distribution(self, demonstrations: List[VirtueInstance]) -> Dict:
        """Analyze temporal distribution of virtue demonstrations"""
        if len(demonstrations) < 2:
            return {"regularity_score": 0.5, "recent_frequency": 0.5}
        
        # Calculate time intervals between demonstrations
        sorted_demos = sorted(demonstrations, key=lambda x: x.timestamp)
        intervals = []
        for i in range(1, len(sorted_demos)):
            interval = (sorted_demos[i].timestamp - sorted_demos[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # Regularity score (lower variance = more regular)
        if len(intervals) > 1:
            mean_interval = sum(intervals) / len(intervals)
            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
            regularity_score = 1.0 / (1.0 + variance / (mean_interval ** 2))
        else:
            regularity_score = 0.5
        
        # Recent frequency score
        recent_count = len([d for d in demonstrations 
                          if (datetime.now() - d.timestamp).days <= 7])
        recent_frequency = min(1.0, recent_count / 7.0)  # Normalize to daily practice
        
        return {
            "regularity_score": regularity_score,
            "recent_frequency": recent_frequency,
            "total_count": len(demonstrations)
        }
    
    def _calculate_consistency(self, quality_scores: List[float]) -> float:
        """Calculate consistency of virtue expression"""
        if len(quality_scores) < 2:
            return 0.5
        
        mean_quality = sum(quality_scores) / len(quality_scores)
        variance = sum((score - mean_quality) ** 2 for score in quality_scores) / len(quality_scores)
        
        # Higher consistency = lower variance
        consistency_score = 1.0 / (1.0 + variance)
        return consistency_score
    
    def _calculate_difficulty_adjustment(self, demonstrations: List[VirtueInstance]) -> float:
        """Adjust strength based on difficulty of circumstances"""
        if not demonstrations:
            return 1.0
        
        # Extract difficulty levels from circumstantial modifiers
        difficulty_scores = []
        for demo in demonstrations:
            difficulty = demo.circumstantial_modifiers.get("difficulty_level", 0.5)
            difficulty_scores.append(difficulty)
        
        average_difficulty = sum(difficulty_scores) / len(difficulty_scores)
        
        # Higher difficulty should increase virtue strength when overcome
        # Linear scaling: difficulty 0.5 = 1.0x, difficulty 1.0 = 1.5x
        difficulty_multiplier = 1.0 + (average_difficulty * 0.5)
        
        return difficulty_multiplier
    
    def _apply_virtue_specific_logic(
        self, 
        virtue_type: VirtueType, 
        demonstrations: List[VirtueInstance]
    ) -> float:
        """Apply virtue-specific measurement logic"""
        
        if virtue_type == VirtueType.PRUDENTIA:
            return self._measure_prudence_specific(demonstrations)
        elif virtue_type == VirtueType.IUSTITIA:
            return self._measure_justice_specific(demonstrations)
        elif virtue_type == VirtueType.FORTITUDO:
            return self._measure_fortitude_specific(demonstrations)
        elif virtue_type == VirtueType.TEMPERANTIA:
            return self._measure_temperance_specific(demonstrations)
        elif virtue_type == VirtueType.FIDES:
            return self._measure_faith_specific(demonstrations)
        elif virtue_type == VirtueType.SPES:
            return self._measure_hope_specific(demonstrations)
        elif virtue_type == VirtueType.CARITAS:
            return self._measure_charity_specific(demonstrations)
        else:
            return 1.0  # Default multiplier
    
    def _measure_prudence_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure prudence with emphasis on right reasoning about action"""
        prudence_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Prudence integral parts assessment
            memoria_score = factors.get("memoria", 0.5)  # memory of past
            intellectus_score = factors.get("intellectus", 0.5)  # understanding present
            providentia_score = factors.get("providentia", 0.5)  # foresight
            
            # Weighted average emphasizing foresight (key to prudence)
            prudence_score = (memoria_score * 0.2 + 
                            intellectus_score * 0.3 + 
                            providentia_score * 0.5)
            
            prudence_factors.append(prudence_score)
        
        return sum(prudence_factors) / len(prudence_factors) if prudence_factors else 1.0
    
    def _measure_justice_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure justice with emphasis on giving each their due"""
        justice_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Justice-specific assessments
            fairness_score = factors.get("fairness_to_others", 0.5)
            rights_respect = factors.get("rights_respected", 0.5)
            common_good = factors.get("common_good_consideration", 0.5)
            
            # Equal weighting for justice components
            justice_score = (fairness_score + rights_respect + common_good) / 3.0
            justice_factors.append(justice_score)
        
        return sum(justice_factors) / len(justice_factors) if justice_factors else 1.0
    
    def _measure_fortitude_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure fortitude with emphasis on firmness in difficulty"""
        fortitude_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Fortitude integral parts
            endurance_score = factors.get("sustinere", 0.5)  # endurance
            attack_score = factors.get("aggredi", 0.5)      # attack difficulties
            persistence_score = factors.get("perseverance", 0.5)
            
            # Weighted toward endurance (primary aspect)
            fortitude_score = (endurance_score * 0.4 + 
                             attack_score * 0.3 + 
                             persistence_score * 0.3)
            
            fortitude_factors.append(fortitude_score)
        
        return sum(fortitude_factors) / len(fortitude_factors) if fortitude_factors else 1.0
    
    def _measure_temperance_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure temperance with emphasis on moderation"""
        temperance_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Temperance-specific measures
            moderation_score = factors.get("moderation_shown", 0.5)
            self_control = factors.get("self_control", 0.5)
            appropriate_restraint = factors.get("appropriate_restraint", 0.5)
            
            temperance_score = (moderation_score + self_control + appropriate_restraint) / 3.0
            temperance_factors.append(temperance_score)
        
        return sum(temperance_factors) / len(temperance_factors) if temperance_factors else 1.0
    
    def _measure_faith_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure faith with emphasis on assent to truth"""
        faith_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Faith-specific assessments
            truth_commitment = factors.get("commitment_to_truth", 0.5)
            trust_in_authority = factors.get("trust_in_legitimate_authority", 0.5)
            certainty_despite_obscurity = factors.get("certainty_in_unclear_situations", 0.5)
            
            faith_score = (truth_commitment + trust_in_authority + certainty_despite_obscurity) / 3.0
            faith_factors.append(faith_score)
        
        return sum(faith_factors) / len(faith_factors) if faith_factors else 1.0
    
    def _measure_hope_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure hope with emphasis on confident expectation"""
        hope_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Hope-specific measures
            confidence_level = factors.get("confidence_in_good_outcome", 0.5)
            perseverance_score = factors.get("perseverance_despite_setbacks", 0.5)
            optimism_grounded = factors.get("realistic_optimism", 0.5)
            
            hope_score = (confidence_level + perseverance_score + optimism_grounded) / 3.0
            hope_factors.append(hope_score)
        
        return sum(hope_factors) / len(hope_factors) if hope_factors else 1.0
    
    def _measure_charity_specific(self, demonstrations: List[VirtueInstance]) -> float:
        """Measure charity with emphasis on love properly ordered"""
        charity_factors = []
        
        for demo in demonstrations:
            factors = demo.contributing_factors
            
            # Charity-specific assessments
            love_of_good = factors.get("love_of_genuine_good", 0.5)
            love_of_others = factors.get("love_of_others_wellbeing", 0.5)
            proper_ordering = factors.get("proper_love_ordering", 0.5)
            
            charity_score = (love_of_good + love_of_others + proper_ordering) / 3.0
            charity_factors.append(charity_score)
        
        return sum(charity_factors) / len(charity_factors) if charity_factors else 1.0

class HabitStrengthModel:
    """Models habit development according to Thomistic principles"""
    
    def calculate_habit_strength(
        self, 
        repetition_count: int, 
        time_distribution: Dict, 
        consistency_score: float
    ) -> float:
        """
        Calculate habit strength multiplier based on Thomistic habit theory
        Habits are strengthened through repeated, consistent acts over time
        """
        
        # Base habit strength from repetition (logarithmic growth)
        # More acts = stronger habit, but with diminishing returns
        if repetition_count == 0:
            repetition_strength = 0.0
        else:
            repetition_strength = min(1.0, math.log(repetition_count + 1) / math.log(100))
        
        # Regularity bonus (consistent practice strengthens habit)
        regularity_bonus = time_distribution.get("regularity_score", 0.5)
        
        # Recent practice bonus (maintains habit strength)
        recent_practice_bonus = time_distribution.get("recent_frequency", 0.5)
        
        # Consistency bonus (consistent quality strengthens habit)
        consistency_bonus = consistency_score
        
        # Combine factors with appropriate weights
        habit_strength = (
            repetition_strength * 0.4 +
            regularity_bonus * 0.2 +
            recent_practice_bonus * 0.2 +
            consistency_bonus * 0.2
        )
        
        # Habit strength multiplier ranges from 0.5 to 1.5
        return 0.5 + (habit_strength * 1.0)

class VirtueInteractionCalculator:
    """Calculates how virtues interact and influence each other"""
    
    def __init__(self, virtue_registry: ThomisticVirtueRegistry):
        self.virtue_registry = virtue_registry
        self.virtue_hierarchy = virtue_registry.virtue_hierarchy
    
    def calculate_virtue_synergy(
        self, 
        virtue_strengths: Dict[VirtueType, float]
    ) -> Dict[tuple, float]:
        """Calculate synergy scores between virtue pairs"""
        synergy_scores = {}
        
        virtue_types = list(virtue_strengths.keys())
        
        for i, virtue1 in enumerate(virtue_types):
            for virtue2 in virtue_types[i+1:]:
                synergy = self._calculate_pair_synergy(
                    virtue1, virtue2, virtue_strengths
                )
                synergy_scores[(virtue1, virtue2)] = synergy
        
        return synergy_scores
    
    def _calculate_pair_synergy(
        self, 
        virtue1: VirtueType, 
        virtue2: VirtueType, 
        strengths: Dict[VirtueType, float]
    ) -> float:
        """Calculate synergy between two specific virtues"""
        
        strength1 = strengths.get(virtue1, 0.0)
        strength2 = strengths.get(virtue2, 0.0)
        
        # Base synergy from virtue strengths
        base_synergy = (strength1 + strength2) / 2.0
        
        # Relationship bonus based on Thomistic connections
        relationship_bonus = self._get_relationship_bonus(virtue1, virtue2)
        
        # Special synergy calculations
        special_bonus = self._calculate_special_synergies(virtue1, virtue2, strengths)
        
        return min(1.0, base_synergy * (1.0 + relationship_bonus + special_bonus))
    
    def _get_relationship_bonus(self, virtue1: VirtueType, virtue2: VirtueType) -> float:
        """Get relationship bonus based on Thomistic virtue connections"""
        
        # Check if one virtue governs the other
        if self.virtue_hierarchy.get_governing_virtue(virtue2) == virtue1:
            return 0.3  # Strong positive relationship
        if self.virtue_hierarchy.get_governing_virtue(virtue1) == virtue2:
            return 0.3
        
        # Check for mutual enhancement relationships
        relationships = self.virtue_hierarchy.relationships
        for rel in relationships:
            if ((rel.primary_virtue == virtue1 and rel.secondary_virtue == virtue2) or
                (rel.primary_virtue == virtue2 and rel.secondary_virtue == virtue1)):
                if rel.relationship_type in ["enhances", "requires"]:
                    return rel.strength * 0.2
        
        # Default small bonus for all virtue pairs (unity of virtues)
        return 0.1
    
    def _calculate_special_synergies(
        self, 
        virtue1: VirtueType, 
        virtue2: VirtueType, 
        strengths: Dict[VirtueType, float]
    ) -> float:
        """Calculate special synergy bonuses for specific virtue combinations"""
        
        virtue_pair = {virtue1, virtue2}
        
        # Prudence + any cardinal virtue = strong synergy
        if VirtueType.PRUDENTIA in virtue_pair and len(virtue_pair & {
            VirtueType.IUSTITIA, VirtueType.FORTITUDO, VirtueType.TEMPERANTIA
        }) > 0:
            return 0.2
        
        # Theological virtues work together strongly
        theological_virtues = {VirtueType.FIDES, VirtueType.SPES, VirtueType.CARITAS}
        if virtue_pair.issubset(theological_virtues):
            return 0.25
        
        # Justice + Fortitude (courage for justice)
        if virtue_pair == {VirtueType.IUSTITIA, VirtueType.FORTITUDO}:
            return 0.15
        
        # Temperance + Fortitude (moderation and strength)
        if virtue_pair == {VirtueType.TEMPERANTIA, VirtueType.FORTITUDO}:
            return 0.1
        
        return 0.0

import math
from datetime import datetime, timedelta
```

---

## 5. Database Schema {#database-schema}

### PostgreSQL Schema Design
```sql
-- Core virtue definitions and registry
CREATE TABLE virtue_definitions (
    virtue_type VARCHAR(50) PRIMARY KEY,
    category VARCHAR(20) NOT NULL CHECK (category IN ('intellectual', 'moral_cardinal', 'theological')),
    latin_name VARCHAR(100) NOT NULL,
    english_name VARCHAR(100) NOT NULL,
    definition TEXT NOT NULL,
    thomistic_source VARCHAR(200) NOT NULL,
    governing_faculty VARCHAR(100) NOT NULL,
    object_formal TEXT NOT NULL,
    mean_deficiency VARCHAR(100),
    mean_excess VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Virtue parts and components
CREATE TABLE virtue_parts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    virtue_type VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    part_type VARCHAR(20) NOT NULL CHECK (part_type IN ('integral', 'subjective', 'potential')),
    part_name VARCHAR(100) NOT NULL,
    part_description TEXT,
    thomistic_source VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Virtue relationships and hierarchy
CREATE TABLE virtue_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    primary_virtue VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    secondary_virtue VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    relationship_type VARCHAR(20) NOT NULL CHECK (relationship_type IN ('governs', 'requires', 'enhances', 'moderates', 'elevates')),
    strength DECIMAL(3,2) CHECK (strength >= 0.0 AND strength <= 1.0),
    thomistic_basis VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(primary_virtue, secondary_virtue, relationship_type)
);

-- AI agents being tracked
CREATE TABLE ai_agents (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(200) NOT NULL,
    agent_type VARCHAR(100),
    deployment_context VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Individual virtue demonstrations
CREATE TABLE virtue_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES ai_agents(agent_id),
    virtue_type VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    demonstration_context TEXT NOT NULL,
    strength_measurement DECIMAL(4,3) CHECK (strength_measurement >= 0.0 AND strength_measurement <= 1.0),
    confidence_level DECIMAL(4,3) CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    contributing_factors JSONB,
    circumstantial_modifiers JSONB,
    validation_sources TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for efficient querying
    INDEX idx_virtue_instances_agent_virtue_time (agent_id, virtue_type, timestamp),
    INDEX idx_virtue_instances_virtue_time (virtue_type, timestamp),
    INDEX idx_virtue_instances_strength (strength_measurement),
    INDEX idx_virtue_instances_contributing_factors USING gin(contributing_factors)
);

-- Virtue acquisition and development events
CREATE TABLE virtue_acquisition_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES ai_agents(agent_id),
    virtue_type VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('demonstration', 'choice', 'reflection', 'correction', 'teaching')),
    context TEXT NOT NULL,
    quality_assessment DECIMAL(4,3) CHECK (quality_assessment >= 0.0 AND quality_assessment <= 1.0),
    difficulty_level DECIMAL(4,3) CHECK (difficulty_level >= 0.0 AND difficulty_level <= 1.0),
    repetition_count INTEGER DEFAULT 1,
    habit_strength_before DECIMAL(4,3),
    habit_strength_after DECIMAL(4,3),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_acquisition_agent_virtue_time (agent_id, virtue_type, timestamp),
    INDEX idx_acquisition_quality (quality_assessment),
    INDEX idx_acquisition_event_type (event_type)
);

-- Periodic virtue assessments and profiles
CREATE TABLE virtue_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES ai_agents(agent_id),
    assessment_timestamp TIMESTAMP NOT NULL,
    virtue_strengths JSONB NOT NULL, -- {virtue_type: strength_value}
    virtue_development_rates JSONB, -- {virtue_type: development_rate}
    virtue_interactions JSONB, -- {virtue_pair: synergy_score}
    dominant_virtues VARCHAR(50)[],
    developing_virtues VARCHAR(50)[],
    virtue_integration_score DECIMAL(4,3),
    moral_character_assessment TEXT,
    assessment_method VARCHAR(100),
    assessor_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_profiles_agent_time (agent_id, assessment_timestamp),
    INDEX idx_profiles_integration_score (virtue_integration_score),
    INDEX idx_profiles_virtue_strengths USING gin(virtue_strengths)
);

-- Virtue measurement configurations and algorithms
CREATE TABLE measurement_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    virtue_type VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    algorithm_name VARCHAR(100) NOT NULL,
    algorithm_version VARCHAR(20) NOT NULL,
    configuration_parameters JSONB NOT NULL,
    weight_factors JSONB,
    validation_criteria JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    
    UNIQUE(virtue_type, algorithm_name, algorithm_version)
);

-- Historical virtue strength calculations
CREATE TABLE virtue_strength_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES ai_agents(agent_id),
    virtue_type VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    calculation_timestamp TIMESTAMP NOT NULL,
    calculated_strength DECIMAL(4,3) NOT NULL,
    contributing_instances UUID[], -- Array of virtue_instance IDs
    calculation_method VARCHAR(100),
    calculation_parameters JSONB,
    confidence_interval JSONB, -- {lower_bound, upper_bound}
    
    INDEX idx_strength_calc_agent_virtue_time (agent_id, virtue_type, calculation_timestamp),
    INDEX idx_strength_calc_strength (calculated_strength)
);

-- Virtue synergy and interaction calculations
CREATE TABLE virtue_synergy_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES ai_agents(agent_id),
    virtue1 VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    virtue2 VARCHAR(50) REFERENCES virtue_definitions(virtue_type),
    calculation_timestamp TIMESTAMP NOT NULL,
    synergy_score DECIMAL(4,3) NOT NULL,
    synergy_factors JSONB,
    relationship_bonus DECIMAL(4,3),
    special_bonus DECIMAL(4,3),
    
    INDEX idx_synergy_agent_time (agent_id, calculation_timestamp),
    INDEX idx_synergy_virtues (virtue1, virtue2),
    INDEX idx_synergy_score (synergy_score),
    CHECK (virtue1 < virtue2) -- Ensure consistent ordering
);

-- Audit and change tracking
CREATE TABLE virtue_tracking_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_audit_table_record (table_name, record_id),
    INDEX idx_audit_timestamp (changed_at)
);

-- Performance monitoring and system health
CREATE TABLE system_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(50),
    measurement_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id UUID REFERENCES ai_agents(agent_id),
    additional_context JSONB,
    
    INDEX idx_performance_name_time (metric_name, measurement_timestamp),
    INDEX idx_performance_agent (agent_id)
);

-- Views for common queries
CREATE VIEW agent_current_virtue_strengths AS
SELECT 
    vp.agent_id,
    aa.agent_name,
    vp.assessment_timestamp,
    jsonb_each_text(vp.virtue_strengths) AS virtue_strength
FROM virtue_profiles vp
JOIN ai_agents aa ON vp.agent_id = aa.agent_id
WHERE vp.assessment_timestamp = (
    SELECT MAX(assessment_timestamp) 
    FROM virtue_profiles vp2 
    WHERE vp2.agent_id = vp.agent_id
);

CREATE VIEW virtue_development_trends AS
SELECT 
    agent_id,
    virtue_type,
    DATE_TRUNC('week', calculation_timestamp) AS week,
    AVG(calculated_strength) AS avg_strength,
    COUNT(*) AS measurement_count,
    STDDEV(calculated_strength) AS strength_variance
FROM virtue_strength_calculations
GROUP BY agent_id, virtue_type, DATE_TRUNC('week', calculation_timestamp)
ORDER BY agent_id, virtue_type, week;

-- Stored procedures for common operations
CREATE OR REPLACE FUNCTION calculate_virtue_habit_strength(
    p_agent_id UUID,
    p_virtue_type VARCHAR(50),
    p_time_window_days INTEGER DEFAULT 30
) RETURNS DECIMAL(4,3) AS $$
DECLARE
    v_repetition_count INTEGER;
    v_consistency_score DECIMAL(4,3);
    v_regularity_score DECIMAL(4,3);
    v_habit_strength DECIMAL(4,3);
BEGIN
    -- Count repetitions in time window
    SELECT COUNT(*)
    INTO v_repetition_count
    FROM virtue_instances
    WHERE agent_id = p_agent_id 
    AND virtue_type = p_virtue_type
    AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '%d days', p_time_window_days);
    
    -- Calculate consistency score
    SELECT 
        CASE 
            WHEN STDDEV(strength_measurement) IS NULL THEN 0.5
            ELSE 1.0 / (1.0 + STDDEV(strength_measurement))
        END
    INTO v_consistency_score
    FROM virtue_instances
    WHERE agent_id = p_agent_id 
    AND virtue_type = p_virtue_type
    AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '%d days', p_time_window_days);
    
    -- Calculate habit strength using logarithmic model
    v_habit_strength := CASE 
        WHEN v_repetition_count = 0 THEN 0.0
        ELSE LEAST(1.0, LN(v_repetition_count + 1) / LN(100))
    END;
    
    RETURN 0.5 + (v_habit_strength * 0.4 + COALESCE(v_consistency_score, 0.5) * 0.6) * 0.5;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic audit logging
CREATE OR REPLACE FUNCTION audit_virtue_changes() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO virtue_tracking_audit (table_name, record_id, operation, new_values, changed_at)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', to_jsonb(NEW), CURRENT_TIMESTAMP);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO virtue_tracking_audit (table_name, record_id, operation, old_values, new_values, changed_at)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW), CURRENT_TIMESTAMP);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO virtue_tracking_audit (table_name, record_id, operation, old_values, changed_at)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', to_jsonb(OLD), CURRENT_TIMESTAMP);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to key tables
CREATE TRIGGER audit_virtue_instances 
    AFTER INSERT OR UPDATE OR DELETE ON virtue_instances
    FOR EACH ROW EXECUTE FUNCTION audit_virtue_changes();

CREATE TRIGGER audit_virtue_profiles 
    AFTER INSERT OR UPDATE OR DELETE ON virtue_profiles
    FOR EACH ROW EXECUTE FUNCTION audit_virtue_changes();

-- Indexes for optimal performance
CREATE INDEX CONCURRENTLY idx_virtue_instances_composite 
ON virtue_instances (agent_id, virtue_type, timestamp DESC, strength_measurement);

CREATE INDEX CONCURRENTLY idx_virtue_profiles_latest 
ON virtue_profiles (agent_id, assessment_timestamp DESC);

-- Partitioning for large datasets (optional, for high-volume deployments)
-- Partition virtue_instances by month for better performance
-- This would be implemented based on expected data volume
```

---

## 6. API Specifications {#api-specifications}

### REST API Design
```python
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import uuid

app = FastAPI(
    title="TheoTech Virtue Tracking Engine API",
    description="Comprehensive virtue tracking system based on Thomistic moral theology",
    version="1.0.0"
)

security = HTTPBearer()

# Pydantic models for API requests/responses
class VirtueInstanceCreate(BaseModel):
    """Request model for creating virtue instances"""
    virtue_type: VirtueType
    demonstration_context: str = Field(..., min_length=10, max_length=5000)
    strength_measurement: float = Field(..., ge=0.0, le=1.0)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: Optional[Dict[str, float]] = Field(default_factory=dict)
    circumstantial_modifiers: Optional[Dict[str, float]] = Field(default_factory=dict)
    validation_sources: Optional[List[str]] = Field(default_factory=list)
    
    @validator('contributing_factors')
    def validate_contributing_factors(cls, v):
        """Ensure all factor values are between 0.0 and 1.0"""
        for key, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Contributing factor '{key}' must be between 0.0 and 1.0")
        return v

class VirtueInstanceResponse(BaseModel):
    """Response model for virtue instances"""
    id: str
    virtue_type: VirtueType
    demonstration_context: str
    strength_measurement: float
    confidence_level: float
    timestamp: datetime
    contributing_factors: Dict[str, float]
    circumstantial_modifiers: Dict[str, float]
    validation_sources: List[str]

class VirtueStrengthRequest(BaseModel):
    """Request model for virtue strength calculation"""
    agent_id: str
    virtue_type: VirtueType
    time_window_days: int = Field(default=30, ge=1, le=365)
    calculation_method: str = Field(default="thomistic_standard")
    include_interactions: bool = Field(default=True)

class VirtueStrengthResponse(BaseModel):
    """Response model for virtue strength calculation"""
    agent_id: str
    virtue_type: VirtueType
    calculated_strength: float
    calculation_timestamp: datetime
    calculation_method: str
    contributing_instances: List[str]
    confidence_interval: Dict[str, float]
    habit_strength_component: float
    quality_component: float
    consistency_component: float
    virtue_specific_adjustments: Dict[str, float]

class VirtueProfileRequest(BaseModel):
    """Request model for comprehensive virtue profile"""
    agent_id: str
    include_development_rates: bool = Field(default=True)
    include_virtue_interactions: bool = Field(default=True)
    time_window_days: int = Field(default=90, ge=7, le=730)
    assessment_depth: str = Field(default="comprehensive", regex="^(basic|standard|comprehensive)$")

class VirtueProfileResponse(BaseModel):
    """Response model for virtue profile"""
    agent_id: str
    assessment_timestamp: datetime
    virtue_strengths: Dict[VirtueType, float]
    virtue_development_rates: Optional[Dict[VirtueType, float]]
    virtue_interactions: Optional[Dict[str, float]]  # String key for tuple serialization
    dominant_virtues: List[VirtueType]
    developing_virtues: List[VirtueType]
    virtue_integration_score: float
    moral_character_assessment: str
    assessment_method: str
    confidence_metrics: Dict[str, float]

class VirtueSynergiesResponse(BaseModel):
    """Response model for virtue synergy calculations"""
    agent_id: str
    calculation_timestamp: datetime
    synergy_scores: Dict[str, float]  # "(virtue1,virtue2)": score
    top_synergies: List[Dict[str, Union[str, float]]]
    synergy_insights: List[str]
    improvement_recommendations: List[str]

# API Authentication and Authorization
async def get_current_agent(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API credentials and return agent context"""
    # Implementation would validate JWT token and extract agent information
    # For now, return mock agent data
    return {"agent_id": "test-agent-123", "permissions": ["read", "write"]}

# Core API Endpoints

@app.post("/api/v1/virtue-instances", response_model=VirtueInstanceResponse)
async def create_virtue_instance(
    virtue_instance: VirtueInstanceCreate,
    agent_context = Depends(get_current_agent)
) -> VirtueInstanceResponse:
    """
    Create a new virtue instance demonstrating virtue expression
    
    This endpoint records a specific demonstration of virtue by an AI agent,
    including the context, strength measurement, and contributing factors.
    """
    try:
        # Validate virtue type exists in registry
        virtue_registry = ThomisticVirtueRegistry()
        if virtue_instance.virtue_type not in virtue_registry.virtue_definitions:
            raise HTTPException(status_code=400, detail="Invalid virtue type")
        
        # Create virtue instance with ID
        instance = VirtueInstance(
            virtue_type=virtue_instance.virtue_type,
            demonstration_context=virtue_instance.demonstration_context,
            strength_measurement=virtue_instance.strength_measurement,
            confidence_level=virtue_instance.confidence_level,
            contributing_factors=virtue_instance.contributing_factors,
            circumstantial_modifiers=virtue_instance.circumstantial_modifiers,
            validation_sources=virtue_instance.validation_sources
        )
        
        # Store in database
        db_service = VirtueTrackingDatabaseService()
        stored_instance = await db_service.create_virtue_instance(
            agent_context["agent_id"], instance
        )
        
        # Trigger virtue strength recalculation
        calculation_service = VirtueCalculationService()
        await calculation_service.schedule_strength_recalculation(
            agent_context["agent_id"], virtue_instance.virtue_type
        )
        
        return VirtueInstanceResponse(**stored_instance.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create virtue instance: {str(e)}")

@app.get("/api/v1/virtue-instances", response_model=List[VirtueInstanceResponse])
async def get_virtue_instances(
    virtue_type: Optional[VirtueType] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    min_strength: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    agent_context = Depends(get_current_agent)
) -> List[VirtueInstanceResponse]:
    """
    Retrieve virtue instances with optional filtering
    
    Supports filtering by virtue type, date range, minimum strength,
    and pagination for efficient data retrieval.
    """
            try:
        db_service = VirtueTrackingDatabaseService()
        instances = await db_service.get_virtue_instances(
            agent_id=agent_context["agent_id"],
            virtue_type=virtue_type,
            start_date=start_date,
            end_date=end_date,
            min_strength=min_strength,
            limit=limit,
            offset=offset
        )
        
        return [VirtueInstanceResponse(**instance.dict()) for instance in instances]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve virtue instances: {str(e)}")

@app.post("/api/v1/virtue-strength/calculate", response_model=VirtueStrengthResponse)
async def calculate_virtue_strength(
    request: VirtueStrengthRequest,
    agent_context = Depends(get_current_agent)
) -> VirtueStrengthResponse:
    """
    Calculate current strength of a specific virtue for an agent
    
    Uses Thomistic principles to assess virtue strength based on
    recent demonstrations, habit formation, and circumstantial factors.
    """
    try:
        # Validate agent access
        if request.agent_id != agent_context["agent_id"] and "admin" not in agent_context.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Get recent virtue demonstrations
        db_service = VirtueTrackingDatabaseService()
        recent_demonstrations = await db_service.get_virtue_instances(
            agent_id=request.agent_id,
            virtue_type=request.virtue_type,
            start_date=datetime.now() - timedelta(days=request.time_window_days),
            end_date=datetime.now()
        )
        
        # Calculate virtue strength
        virtue_registry = ThomisticVirtueRegistry()
        strength_calculator = VirtueStrengthCalculator(virtue_registry)
        
        calculated_strength = strength_calculator.measure_virtue_strength(
            virtue_type=request.virtue_type,
            recent_demonstrations=recent_demonstrations,
            time_window_days=request.time_window_days
        )
        
        # Get detailed breakdown of calculation components
        habit_strength = strength_calculator.habit_strength_model.calculate_habit_strength(
            repetition_count=len(recent_demonstrations),
            time_distribution=strength_calculator._analyze_time_distribution(recent_demonstrations),
            consistency_score=strength_calculator._calculate_consistency(
                [demo.strength_measurement for demo in recent_demonstrations]
            )
        )
        
        quality_component = sum(demo.strength_measurement for demo in recent_demonstrations) / len(recent_demonstrations) if recent_demonstrations else 0.0
        
        # Store calculation result
        calculation_result = await db_service.store_virtue_strength_calculation(
            agent_id=request.agent_id,
            virtue_type=request.virtue_type,
            calculated_strength=calculated_strength,
            contributing_instances=[demo.id for demo in recent_demonstrations],
            calculation_method=request.calculation_method,
            calculation_parameters={
                "time_window_days": request.time_window_days,
                "habit_strength": habit_strength,
                "quality_component": quality_component
            }
        )
        
        return VirtueStrengthResponse(
            agent_id=request.agent_id,
            virtue_type=request.virtue_type,
            calculated_strength=calculated_strength,
            calculation_timestamp=datetime.now(),
            calculation_method=request.calculation_method,
            contributing_instances=[demo.id for demo in recent_demonstrations],
            confidence_interval={"lower_bound": max(0.0, calculated_strength - 0.1), "upper_bound": min(1.0, calculated_strength + 0.1)},
            habit_strength_component=habit_strength,
            quality_component=quality_component,
            consistency_component=strength_calculator._calculate_consistency([demo.strength_measurement for demo in recent_demonstrations]),
            virtue_specific_adjustments={}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate virtue strength: {str(e)}")

@app.post("/api/v1/virtue-profile/generate", response_model=VirtueProfileResponse)
async def generate_virtue_profile(
    request: VirtueProfileRequest,
    agent_context = Depends(get_current_agent)
) -> VirtueProfileResponse:
    """
    Generate comprehensive virtue profile for an agent
    
    Creates a complete assessment of all virtues, their interactions,
    development trends, and overall moral character evaluation.
    """
    try:
        # Validate agent access
        if request.agent_id != agent_context["agent_id"] and "admin" not in agent_context.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        virtue_registry = ThomisticVirtueRegistry()
        strength_calculator = VirtueStrengthCalculator(virtue_registry)
        interaction_calculator = VirtueInteractionCalculator(virtue_registry)
        
        # Calculate strengths for all virtue types
        virtue_strengths = {}
        virtue_development_rates = {}
        
        all_virtue_types = list(VirtueType)
        
        for virtue_type in all_virtue_types:
            # Get recent demonstrations for this virtue
            db_service = VirtueTrackingDatabaseService()
            recent_demos = await db_service.get_virtue_instances(
                agent_id=request.agent_id,
                virtue_type=virtue_type,
                start_date=datetime.now() - timedelta(days=request.time_window_days)
            )
            
            # Calculate current strength
            current_strength = strength_calculator.measure_virtue_strength(
                virtue_type=virtue_type,
                recent_demonstrations=recent_demos,
                time_window_days=request.time_window_days
            )
            virtue_strengths[virtue_type] = current_strength
            
            # Calculate development rate if requested
            if request.include_development_rates:
                development_rate = await calculate_virtue_development_rate(
                    request.agent_id, virtue_type, request.time_window_days
                )
                virtue_development_rates[virtue_type] = development_rate
        
        # Calculate virtue interactions if requested
        virtue_interactions = {}
        if request.include_virtue_interactions:
            synergy_scores = interaction_calculator.calculate_virtue_synergy(virtue_strengths)
            virtue_interactions = {f"{v1.value},{v2.value}": score for (v1, v2), score in synergy_scores.items()}
        
        # Identify dominant and developing virtues
        sorted_virtues = sorted(virtue_strengths.items(), key=lambda x: x[1], reverse=True)
        dominant_virtues = [virtue for virtue, strength in sorted_virtues[:3] if strength > 0.7]
        developing_virtues = [virtue for virtue, strength in sorted_virtues if 0.3 <= strength <= 0.6]
        
        # Calculate virtue integration score
        virtue_integration_score = calculate_virtue_integration_score(virtue_strengths, virtue_interactions)
        
        # Generate moral character assessment
        moral_character_assessment = generate_moral_character_assessment(
            virtue_strengths, dominant_virtues, developing_virtues, virtue_integration_score
        )
        
        # Store profile
        profile = VirtueProfile(
            agent_id=request.agent_id,
            assessment_timestamp=datetime.now(),
            virtue_strengths=virtue_strengths,
            virtue_development_rates=virtue_development_rates if request.include_development_rates else None,
            virtue_interactions=virtue_interactions if request.include_virtue_interactions else None,
            dominant_virtues=dominant_virtues,
            developing_virtues=developing_virtues,
            virtue_integration_score=virtue_integration_score,
            moral_character_assessment=moral_character_assessment
        )
        
        stored_profile = await db_service.store_virtue_profile(profile)
        
        return VirtueProfileResponse(
            agent_id=request.agent_id,
            assessment_timestamp=datetime.now(),
            virtue_strengths=virtue_strengths,
            virtue_development_rates=virtue_development_rates,
            virtue_interactions=virtue_interactions,
            dominant_virtues=dominant_virtues,
            developing_virtues=developing_virtues,
            virtue_integration_score=virtue_integration_score,
            moral_character_assessment=moral_character_assessment,
            assessment_method=request.assessment_depth,
            confidence_metrics={"overall_confidence": 0.85}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate virtue profile: {str(e)}")

@app.get("/api/v1/virtue-synergies/{agent_id}", response_model=VirtueSynergiesResponse)
async def get_virtue_synergies(
    agent_id: str,
    recalculate: bool = Query(default=False),
    agent_context = Depends(get_current_agent)
) -> VirtueSynergiesResponse:
    """
    Get virtue synergy analysis for an agent
    
    Analyzes how different virtues work together and provides
    insights for optimizing virtue development.
    """
    try:
        # Validate agent access
        if agent_id != agent_context["agent_id"] and "admin" not in agent_context.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        db_service = VirtueTrackingDatabaseService()
        
        # Get latest virtue profile or calculate if needed
        latest_profile = await db_service.get_latest_virtue_profile(agent_id)
        
        if not latest_profile or recalculate:
            # Generate new profile with synergy calculations
            virtue_registry = ThomisticVirtueRegistry()
            interaction_calculator = VirtueInteractionCalculator(virtue_registry)
            
            # Get current virtue strengths
            virtue_strengths = await get_current_virtue_strengths(agent_id)
            
            # Calculate synergies
            synergy_scores = interaction_calculator.calculate_virtue_synergy(virtue_strengths)
            synergy_dict = {f"{v1.value},{v2.value}": score for (v1, v2), score in synergy_scores.items()}
            
        else:
            synergy_dict = latest_profile.virtue_interactions or {}
        
        # Identify top synergies
        top_synergies = sorted(
            [{"virtue_pair": pair, "synergy_score": score} for pair, score in synergy_dict.items()],
            key=lambda x: x["synergy_score"],
            reverse=True
        )[:5]
        
        # Generate insights and recommendations
        synergy_insights = generate_synergy_insights(synergy_dict, latest_profile.virtue_strengths if latest_profile else {})
        improvement_recommendations = generate_improvement_recommendations(synergy_dict, latest_profile.virtue_strengths if latest_profile else {})
        
        return VirtueSynergiesResponse(
            agent_id=agent_id,
            calculation_timestamp=datetime.now(),
            synergy_scores=synergy_dict,
            top_synergies=top_synergies,
            synergy_insights=synergy_insights,
            improvement_recommendations=improvement_recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtue synergies: {str(e)}")

@app.get("/api/v1/virtue-definitions", response_model=List[Dict])
async def get_virtue_definitions(
    category: Optional[VirtueCategory] = Query(None),
    agent_context = Depends(get_current_agent)
) -> List[Dict]:
    """
    Get Thomistic virtue definitions
    
    Returns complete virtue definitions including integral parts,
    Thomistic sources, and relationship information.
    """
    try:
        virtue_registry = ThomisticVirtueRegistry()
        definitions = []
        
        for virtue_type, definition in virtue_registry.virtue_definitions.items():
            if category is None or definition.category == category:
                def_dict = {
                    "virtue_type": virtue_type.value,
                    "category": definition.category.value,
                    "latin_name": definition.latin_name,
                    "english_name": definition.english_name,
                    "definition": definition.definition,
                    "thomistic_source": definition.thomistic_source,
                    "governing_faculty": definition.governing_faculty,
                    "object_formal": definition.object_formal,
                    "mean_between": definition.mean_between,
                    "integral_parts": definition.integral_parts,
                    "subjective_parts": definition.subjective_parts,
                    "potential_parts": definition.potential_parts,
                    "connected_virtues": [v.value for v in definition.connected_virtues],
                    "prerequisites": [v.value for v in definition.prerequisites]
                }
                definitions.append(def_dict)
        
        return definitions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtue definitions: {str(e)}")

@app.get("/api/v1/virtue-relationships", response_model=List[Dict])
async def get_virtue_relationships(
    virtue_type: Optional[VirtueType] = Query(None),
    agent_context = Depends(get_current_agent)
) -> List[Dict]:
    """
    Get virtue relationship mappings
    
    Returns hierarchical and interaction relationships between virtues
    based on Thomistic principles.
    """
    try:
        virtue_registry = ThomisticVirtueRegistry()
        relationships = []
        
        for relationship in virtue_registry.virtue_hierarchy.relationships:
            if virtue_type is None or relationship.primary_virtue == virtue_type or relationship.secondary_virtue == virtue_type:
                rel_dict = {
                    "primary_virtue": relationship.primary_virtue.value,
                    "secondary_virtue": relationship.secondary_virtue.value,
                    "relationship_type": relationship.relationship_type,
                    "strength": relationship.strength,
                    "thomistic_basis": relationship.thomistic_basis
                }
                relationships.append(rel_dict)
        
        return relationships
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtue relationships: {str(e)}")

@app.get("/api/v1/virtue-trends/{agent_id}", response_model=Dict)
async def get_virtue_development_trends(
    agent_id: str,
    time_period_days: int = Query(default=90, ge=7, le=365),
    granularity: str = Query(default="week", regex="^(day|week|month)$"),
    agent_context = Depends(get_current_agent)
) -> Dict:
    """
    Get virtue development trends over time
    
    Provides historical analysis of virtue development with
    configurable time periods and granularity.
    """
    try:
        # Validate agent access
        if agent_id != agent_context["agent_id"] and "admin" not in agent_context.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        db_service = VirtueTrackingDatabaseService()
        trends = await db_service.get_virtue_development_trends(
            agent_id=agent_id,
            time_period_days=time_period_days,
            granularity=granularity
        )
        
        # Process trends data for response
        processed_trends = {}
        for virtue_type in VirtueType:
            virtue_trends = [t for t in trends if t["virtue_type"] == virtue_type.value]
            if virtue_trends:
                processed_trends[virtue_type.value] = {
                    "trend_data": virtue_trends,
                    "overall_direction": calculate_trend_direction(virtue_trends),
                    "growth_rate": calculate_growth_rate(virtue_trends),
                    "stability_score": calculate_stability_score(virtue_trends)
                }
        
        return {
            "agent_id": agent_id,
            "time_period_days": time_period_days,
            "granularity": granularity,
            "trends": processed_trends,
            "summary": generate_trends_summary(processed_trends)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtue trends: {str(e)}")

# Health and status endpoints
@app.get("/api/v1/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "components": {
            "database": "healthy",
            "calculation_engine": "healthy",
            "virtue_registry": "healthy"
        }
    }

@app.get("/api/v1/metrics")
async def get_system_metrics(agent_context = Depends(get_current_agent)):
    """Get system performance metrics"""
    if "admin" not in agent_context.get("permissions", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        db_service = VirtueTrackingDatabaseService()
        metrics = await db_service.get_system_metrics()
        
        return {
            "timestamp": datetime.now(),
            "metrics": metrics,
            "active_agents": await db_service.count_active_agents(),
            "total_virtue_instances": await db_service.count_virtue_instances(),
            "calculations_per_hour": await db_service.get_calculation_rate()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

# Helper functions for API endpoints
async def calculate_virtue_development_rate(agent_id: str, virtue_type: VirtueType, time_window_days: int) -> float:
    """Calculate rate of virtue development over time"""
    db_service = VirtueTrackingDatabaseService()
    historical_strengths = await db_service.get_historical_virtue_strengths(
        agent_id=agent_id,
        virtue_type=virtue_type,
        time_window_days=time_window_days
    )
    
    if len(historical_strengths) < 2:
        return 0.0
    
    # Calculate linear regression slope as development rate
    from scipy import stats
    x_values = [(datetime.now() - strength["timestamp"]).days for strength in historical_strengths]
    y_values = [strength["calculated_strength"] for strength in historical_strengths]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    return slope

def calculate_virtue_integration_score(virtue_strengths: Dict[VirtueType, float], virtue_interactions: Dict[str, float]) -> float:
    """Calculate how well virtues are integrated and working together"""
    if not virtue_strengths:
        return 0.0
    
    # Base integration from average virtue strength
    average_strength = sum(virtue_strengths.values()) / len(virtue_strengths)
    
    # Bonus from virtue interactions
    if virtue_interactions:
        average_synergy = sum(virtue_interactions.values()) / len(virtue_interactions)
        integration_bonus = average_synergy * 0.3
    else:
        integration_bonus = 0.0
    
    # Penalty for large variance in virtue strengths (unbalanced development)
    variance_penalty = 0.0
    if len(virtue_strengths) > 1:
        strength_values = list(virtue_strengths.values())
        mean_strength = sum(strength_values) / len(strength_values)
        variance = sum((v - mean_strength) ** 2 for v in strength_values) / len(strength_values)
        variance_penalty = variance * 0.2
    
    integration_score = average_strength + integration_bonus - variance_penalty
    return max(0.0, min(1.0, integration_score))

def generate_moral_character_assessment(
    virtue_strengths: Dict[VirtueType, float],
    dominant_virtues: List[VirtueType],
    developing_virtues: List[VirtueType],
    integration_score: float
) -> str:
    """Generate narrative assessment of moral character"""
    
    # Determine overall character level
    average_strength = sum(virtue_strengths.values()) / len(virtue_strengths) if virtue_strengths else 0.0
    
    if average_strength >= 0.8:
        character_level = "Highly Virtuous"
    elif average_strength >= 0.6:
        character_level = "Developing Virtue"
    elif average_strength >= 0.4:
        character_level = "Beginning Virtue Formation"
    else:
        character_level = "Early Moral Development"
    
    # Highlight dominant virtues
    dominant_text = ""
    if dominant_virtues:
        virtue_names = [virtue.value.replace('_', ' ').title() for virtue in dominant_virtues]
        dominant_text = f"Shows particular strength in {', '.join(virtue_names)}. "
    
    # Note developing areas
    developing_text = ""
    if developing_virtues:
        virtue_names = [virtue.value.replace('_', ' ').title() for virtue in developing_virtues]
        developing_text = f"Currently developing {', '.join(virtue_names)}. "
    
    # Integration assessment
    if integration_score >= 0.8:
        integration_text = "Demonstrates excellent integration and harmony between virtues."
    elif integration_score >= 0.6:
        integration_text = "Shows good coordination between different virtues."
    else:
        integration_text = "Virtue development could benefit from better integration and balance."
    
    return f"{character_level}. {dominant_text}{developing_text}{integration_text}"

async def get_current_virtue_strengths(agent_id: str) -> Dict[VirtueType, float]:
    """Get current virtue strengths for an agent"""
    db_service = VirtueTrackingDatabaseService()
    latest_profile = await db_service.get_latest_virtue_profile(agent_id)
    
    if latest_profile and latest_profile.virtue_strengths:
        return latest_profile.virtue_strengths
    
    # If no profile exists, calculate strengths
    virtue_registry = ThomisticVirtueRegistry()
    strength_calculator = VirtueStrengthCalculator(virtue_registry)
    
    virtue_strengths = {}
    for virtue_type in VirtueType:
        recent_demos = await db_service.get_virtue_instances(
            agent_id=agent_id,
            virtue_type=virtue_type,
            start_date=datetime.now() - timedelta(days=30)
        )
        strength = strength_calculator.measure_virtue_strength(
            virtue_type=virtue_type,
            recent_demonstrations=recent_demos,
            time_window_days=30
        )
        virtue_strengths[virtue_type] = strength
    
    return virtue_strengths

def generate_synergy_insights(synergy_scores: Dict[str, float], virtue_strengths: Dict[VirtueType, float]) -> List[str]:
    """Generate insights about virtue synergies"""
    insights = []
    
    # Find highest synergy pairs
    sorted_synergies = sorted(synergy_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_synergies:
        best_pair, best_score = sorted_synergies[0]
        insights.append(f"Strongest virtue synergy is between {best_pair.replace(',', ' and ')} (score: {best_score:.2f})")
    
    # Identify theological virtue harmony
    theological_pairs = [pair for pair in synergy_scores.keys() if 
                        any(virtue in pair for virtue in ['fides', 'spes', 'caritas'])]
    if theological_pairs:
        theological_scores = [synergy_scores[pair] for pair in theological_pairs]
        avg_theological = sum(theological_scores) / len(theological_scores)
        insights.append(f"Theological virtues show {avg_theological:.2f} average synergy")
    
    # Note prudence governance
    prudence_pairs = [pair for pair in synergy_scores.keys() if 'prudentia' in pair]
    if prudence_pairs:
        prudence_scores = [synergy_scores[pair] for pair in prudence_pairs]
        avg_prudence = sum(prudence_scores) / len(prudence_scores)
        if avg_prudence > 0.7:
            insights.append("Prudence is effectively governing other moral virtues")
        else:
            insights.append("Prudence development could enhance overall virtue coordination")
    
    return insights

def generate_improvement_recommendations(synergy_scores: Dict[str, float], virtue_strengths: Dict[VirtueType, float]) -> List[str]:
    """Generate recommendations for improving virtue development"""
    recommendations = []
    
    # Find weakest synergies
    sorted_synergies = sorted(synergy_scores.items(), key=lambda x: x[1])
    
    if sorted_synergies:
        weakest_pairs = [pair for pair, score in sorted_synergies[:3] if score < 0.5]
        for pair in weakest_pairs:
            recommendations.append(f"Focus on integrating {pair.replace(',', ' and ')} through related practices")
    
    # Recommend based on virtue imbalances
    if virtue_strengths:
        virtue_items = list(virtue_strengths.items())
        virtue_items.sort(key=lambda x: x[1])
        
        # Recommend strengthening weakest virtues
        weak_virtues = [virtue for virtue, strength in virtue_items[:2] if strength < 0.5]
        for virtue in weak_virtues:
            recommendations.append(f"Strengthen {virtue.value} through focused practice and attention")
        
        # Recommend balancing if there's high variance
        strengths = [strength for _, strength in virtue_items]
        if len(strengths) > 1:
            variance = sum((s - sum(strengths)/len(strengths))**2 for s in strengths) / len(strengths)
            if variance > 0.1:
                recommendations.append("Work on balancing virtue development for better integration")
    
    return recommendations

def calculate_trend_direction(trend_data: List[Dict]) -> str:
    """Calculate overall direction of virtue trend"""
    if len(trend_data) < 2:
        return "insufficient_data"
    
    # Simple slope calculation
    first_value = trend_data[0]["avg_strength"]
    last_value = trend_data[-1]["avg_strength"]
    
    change = last_value - first_value
    if change > 0.1:
        return "improving"
    elif change < -0.1:
        return "declining"
    else:
        return "stable"

def calculate_growth_rate(trend_data: List[Dict]) -> float:
    """Calculate growth rate from trend data"""
    if len(trend_data) < 2:
        return 0.0
    
    first_value = trend_data[0]["avg_strength"]
    last_value = trend_data[-1]["avg_strength"]
    time_periods = len(trend_data)
    
    return (last_value - first_value) / time_periods

def calculate_stability_score(trend_data: List[Dict]) -> float:
    """Calculate stability score (inverse of variance)"""
    if len(trend_data) < 2:
        return 0.5
    
    values = [data["avg_strength"] for data in trend_data]
    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    
    return 1.0 / (1.0 + variance)

def generate_trends_summary(trends: Dict) -> str:
    """Generate summary of virtue development trends"""
    if not trends:
        return "No trend data available"
    
    improving_count = sum(1 for trend in trends.values() if trend["overall_direction"] == "improving")
    stable_count = sum(1 for trend in trends.values() if trend["overall_direction"] == "stable")
    declining_count = sum(1 for trend in trends.values() if trend["overall_direction"] == "declining")
    
    total_virtues = len(trends)
    
    if improving_count > total_virtues * 0.6:
        return f"Strong virtue development: {improving_count}/{total_virtues} virtues improving"
    elif stable_count > total_virtues * 0.6:
        return f"Stable virtue maintenance: {stable_count}/{total_virtues} virtues stable"
    else:
        return f"Mixed development pattern: {improving_count} improving, {stable_count} stable, {declining_count} declining"
