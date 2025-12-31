const { useState, useEffect, useRef, useCallback } = React;

// Speaker colors for timeline
const SPEAKER_COLORS = [
    '#ff416c', '#667eea', '#11998e', '#f7971e', '#8e44ad',
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'
];

function getSpeakerColor(speaker, speakerMap) {
    if (!speaker) return 'rgba(128, 128, 128, 0.5)';
    if (!speakerMap.has(speaker)) {
        speakerMap.set(speaker, SPEAKER_COLORS[speakerMap.size % SPEAKER_COLORS.length]);
    }
    return speakerMap.get(speaker);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
}

function VoiceRecorder() {
    const [isRecording, setIsRecording] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [status, setStatus] = useState('ready');
    const [statusText, setStatusText] = useState('Ready to record');
    const [audioBlob, setAudioBlob] = useState(null);
    const [duration, setDuration] = useState(0);
    const [segments, setSegments] = useState([]);
    const [refinedSegments, setRefinedSegments] = useState([]); // TS-VAD refined segments
    const [events, setEvents] = useState([]);
    const [speakers, setSpeakers] = useState([]);
    const [speakerMap] = useState(new Map());
    const [activeSegment, setActiveSegment] = useState(null);
    const [namingSegment, setNamingSegment] = useState(null);
    const [newSpeakerName, setNewSpeakerName] = useState('');
    const [selectedExistingSpeaker, setSelectedExistingSpeaker] = useState(null);
    const [isSaving, setIsSaving] = useState(false);
    const [knownSpeakers, setKnownSpeakers] = useState([]);
    const [unknownSpeakers, setUnknownSpeakers] = useState([]); // unknown speaker clusters from TS-VAD
    
    // Saved recordings state
    const [savedRecordings, setSavedRecordings] = useState([]);
    const [recordingName, setRecordingName] = useState('');
    const [currentRecordingId, setCurrentRecordingId] = useState(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [selection, setSelection] = useState(null); // {start, end} for drag selection
    const [isSavingRecording, setIsSavingRecording] = useState(false);
    const [isLoadingRecording, setIsLoadingRecording] = useState(false);
    
    // Speaker comparison state
    const [compareSpeaker, setCompareSpeaker] = useState(null);
    const [segmentSimilarities, setSegmentSimilarities] = useState({}); // {"start-end": similarity}
    const [identifyingSegment, setIdentifyingSegment] = useState(null); // segment being identified
    const [segmentIdentifications, setSegmentIdentifications] = useState({}); // {"start-end": {matches, top_match}}

    const waveformRef = useRef(null);
    const playbackWaveformRef = useRef(null);
    const wavesurferRef = useRef(null);
    const playbackWavesurferRef = useRef(null);
    const regionsRef = useRef(null);
    const selectionRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const streamRef = useRef(null);
    const processingRef = useRef(false);
    const lastProcessedChunkRef = useRef(0);
    const activeSegmentRef = useRef(null);
    const canvasRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const animationRef = useRef(null);

    const pickSupportedAudioMimeType = useCallback(() => {
        if (!window.MediaRecorder || typeof MediaRecorder.isTypeSupported !== 'function') {
            return null;
        }

        const candidates = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
        ];

        for (const candidate of candidates) {
            try {
                if (MediaRecorder.isTypeSupported(candidate)) return candidate;
            } catch (_) {
                // ignore
            }
        }

        return null;
    }, []);

    const filenameForBlob = useCallback((blob) => {
        const type = (blob && blob.type) ? blob.type : '';
        if (type.includes('webm')) return 'recording.webm';
        if (type.includes('mp4')) return 'recording.mp4';
        return 'recording.audio';
    }, []);

    // Initialize recording waveform
    useEffect(() => {
        if (waveformRef.current && !wavesurferRef.current) {
            wavesurferRef.current = WaveSurfer.create({
                container: waveformRef.current,
                waveColor: '#4a9eff',
                progressColor: '#00d2ff',
                cursorColor: '#fff',
                barWidth: 2,
                barGap: 1,
                barRadius: 2,
                height: 50,
                normalize: true,
            });
        }

        return () => {
            if (wavesurferRef.current) {
                wavesurferRef.current.destroy();
                wavesurferRef.current = null;
            }
        };
    }, []);

    // Initialize playback waveform when audio is available
    useEffect(() => {
        if (playbackWaveformRef.current && audioBlob && !playbackWavesurferRef.current) {
            // Create regions plugin for drag selection
            const regions = WaveSurfer.Regions.create();
            regionsRef.current = regions;

            playbackWavesurferRef.current = WaveSurfer.create({
                container: playbackWaveformRef.current,
                waveColor: '#4a9eff',
                progressColor: '#00d2ff',
                cursorColor: '#fff',
                barWidth: 2,
                barGap: 1,
                barRadius: 2,
                height: 50,
                normalize: true,
                plugins: [regions],
            });

            // Enable drag selection to create regions
            regions.enableDragSelection({
                color: 'rgba(0, 210, 255, 0.3)',
            });

            // When a region is created or updated
            regions.on('region-created', (region) => {
                // Remove any existing regions first (only allow one selection)
                regions.getRegions().forEach(r => {
                    if (r.id !== region.id) r.remove();
                });
                const sel = { start: region.start, end: region.end };
                setSelection(sel);
                selectionRef.current = sel;
            });

            regions.on('region-updated', (region) => {
                const sel = { start: region.start, end: region.end };
                setSelection(sel);
                selectionRef.current = sel;
            });

            // Double-click on region to play it
            regions.on('region-double-clicked', (region) => {
                region.play();
            });

            const url = URL.createObjectURL(audioBlob);
            playbackWavesurferRef.current.load(url);

            // Double-click on waveform (outside region) clears selection
            playbackWavesurferRef.current.on('dblclick', () => {
                if (regionsRef.current) {
                    regionsRef.current.clearRegions();
                }
                setSelection(null);
                selectionRef.current = null;
            });

            playbackWavesurferRef.current.on('finish', () => {
                setIsPlaying(false);
            });

            playbackWavesurferRef.current.on('play', () => {
                setIsPlaying(true);
            });

            playbackWavesurferRef.current.on('pause', () => {
                setIsPlaying(false);
            });

            // Listen for timeupdate to track position and stop at segment/selection end
            playbackWavesurferRef.current.on('timeupdate', (time) => {
                setCurrentTime(time);
                // Stop at segment end if playing a segment
                if (activeSegmentRef.current && time >= activeSegmentRef.current.end) {
                    playbackWavesurferRef.current.pause();
                    setActiveSegment(null);
                    activeSegmentRef.current = null;
                }
                // Stop at selection end if playing within a selection
                if (selectionRef.current && time >= selectionRef.current.end) {
                    playbackWavesurferRef.current.pause();
                    // Seek back to selection start for next play
                    const position = selectionRef.current.start / playbackWavesurferRef.current.getDuration();
                    playbackWavesurferRef.current.seekTo(position);
                }
            });
        }

        return () => {
            if (playbackWavesurferRef.current) {
                playbackWavesurferRef.current.destroy();
                playbackWavesurferRef.current = null;
                regionsRef.current = null;
                setSelection(null);
                selectionRef.current = null;
            }
        };
    }, [audioBlob]);

    const startRecording = useCallback(async () => {
        try {
            // Note: Don't specify sampleRate - browsers don't support it well
            // The backend will resample to 16kHz via ffmpeg
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            streamRef.current = stream;
            console.log('[Recording] Got media stream:', stream.getAudioTracks()[0]?.getSettings());

            // Clear previous state
            audioChunksRef.current = [];
            lastProcessedChunkRef.current = 0;
            processingRef.current = false;
            setSegments([]);
            setRefinedSegments([]);
            setUnknownSpeakers([]);
            setEvents([]);
            setSpeakers([]);
            speakerMap.clear();
            setActiveSegment(null);
            activeSegmentRef.current = null;
            setCurrentRecordingId(null);
            
            // Reset URL to home
            if (window.location.pathname !== '/') {
                window.history.pushState({}, '', '/');
            }

            // Cleanup previous playback wavesurfer
            if (playbackWavesurferRef.current) {
                playbackWavesurferRef.current.destroy();
                playbackWavesurferRef.current = null;
            }
            setAudioBlob(null);

            // Create MediaRecorder
            const mimeType = pickSupportedAudioMimeType();
            console.log('[Recording] Using mimeType:', mimeType);
            const recorderOptions = mimeType ? { mimeType } : undefined;
            mediaRecorderRef.current = new MediaRecorder(stream, recorderOptions);
            console.log('[Recording] MediaRecorder created, state:', mediaRecorderRef.current.state);

            mediaRecorderRef.current.ondataavailable = async (event) => {
                console.log('[Recording] ondataavailable, size:', event.data.size, 'type:', event.data.type);
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                    console.log('[Recording] Total chunks:', audioChunksRef.current.length);
                    
                    // Stream to backend every 3 chunks (3 seconds) for real-time results
                    if (audioChunksRef.current.length >= 3 && 
                        audioChunksRef.current.length % 2 === 0 &&
                        !processingRef.current) {
                        const blob = new Blob(audioChunksRef.current, { type: mimeType || event.data.type });
                        console.log('[Recording] Sending partial blob, size:', blob.size);
                        // Process in background, don't await
                        processStreamingDiarization(blob, true);
                    }
                }
            };

            mediaRecorderRef.current.onstop = async () => {
                console.log('[Recording] onstop, total chunks:', audioChunksRef.current.length);
                const blobType = mimeType || (audioChunksRef.current[0] && audioChunksRef.current[0].type) || 'audio/webm';
                const blob = new Blob(audioChunksRef.current, { type: blobType });
                console.log('[Recording] Final blob size:', blob.size, 'type:', blob.type);
                setAudioBlob(blob);

                try {
                    // Final diarization with complete audio (coarse streaming)
                    await processStreamingDiarization(blob, false);

                    //Run TS-VAD refinement pass (optional - comment out if causing issues)
                    await processRefinedDiarization(blob);
                } finally {
                    if (streamRef.current) {
                        streamRef.current.getTracks().forEach(track => track.stop());
                        streamRef.current = null;
                    }
                    if (audioContextRef.current) {
                        audioContextRef.current.close();
                        audioContextRef.current = null;
                    }
                }
            };

            // Start live visualization using AudioContext BEFORE starting MediaRecorder
            // This ensures the analyser is ready when recording begins
            const audioContext = new AudioContext();
            audioContextRef.current = audioContext;
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            console.log('[Recording] AudioContext state:', audioContext.state);
            
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyserRef.current = analyser;
            analyser.fftSize = 2048;
            source.connect(analyser);

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            // Start the waveform animation loop
            const drawWaveform = () => {
                // Continue drawing even if MediaRecorder hasn't started yet
                if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') {
                    // Keep trying until recording starts, or we're stopped
                    if (streamRef.current) {
                        animationRef.current = requestAnimationFrame(drawWaveform);
                    }
                    return;
                }
                
                if (mediaRecorderRef.current.state !== 'recording' && mediaRecorderRef.current.state !== 'paused') {
                    animationRef.current = null;
                    return;
                }
                
                animationRef.current = requestAnimationFrame(drawWaveform);
                
                const canvas = canvasRef.current;
                if (!canvas) return;
                
                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;
                
                analyser.getByteTimeDomainData(dataArray);
                
                // Clear canvas
                ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.fillRect(0, 0, width, height);
                
                // Draw waveform
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#4a9eff';
                ctx.beginPath();
                
                const sliceWidth = width / bufferLength;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = (v * height) / 2;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                    x += sliceWidth;
                }
                
                ctx.lineTo(width, height / 2);
                ctx.stroke();
                
                // Draw center line
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.stroke();
                
                // Add glow effect based on intensity
                const average = dataArray.reduce((a, b) => a + Math.abs(b - 128), 0) / bufferLength;
                const intensity = Math.min(average / 50, 1);
                if (waveformRef.current) {
                    waveformRef.current.style.boxShadow = `inset 0 0 ${20 + intensity * 30}px rgba(74, 158, 255, ${intensity * 0.5})`;
                }
            };
            
            // Start waveform animation immediately
            drawWaveform();

            // Now start recording with timeslice for streaming (1 second chunks)
            mediaRecorderRef.current.start(1000);
            console.log('[Recording] MediaRecorder started, state:', mediaRecorderRef.current.state);
            setIsRecording(true);
            setIsPaused(false);
            setStatus('recording');
            setStatusText('Recording... Results will appear in real-time');

        } catch (error) {
            console.error('Error starting recording:', error);
            setStatus('ready');
            setStatusText('Error: Could not access microphone');
        }
    }, [speakerMap, processStreamingDiarization, processRefinedDiarization, pickSupportedAudioMimeType]);

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && (mediaRecorderRef.current.state === 'recording' || mediaRecorderRef.current.state === 'paused')) {
            mediaRecorderRef.current.stop();
        }
        // Cancel animation frame
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }
        setIsRecording(false);
        setIsPaused(false);
        setStatus('processing');
        setStatusText('Finalizing analysis...');
        
        // Reset waveform glow
        if (waveformRef.current) {
            waveformRef.current.style.boxShadow = 'none';
        }
    }, []);

    const pauseRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.pause();
            setIsPaused(true);
            setStatusText('Recording paused');
        }
    }, []);

    const resumeRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'paused') {
            mediaRecorderRef.current.resume();
            setIsPaused(false);
            setStatusText('Recording... Results will appear in real-time');
        }
    }, []);

    const processStreamingDiarization = useCallback(async (blob, isPartial = false) => {
        if (processingRef.current) return;
        processingRef.current = true;

        try {
            const formData = new FormData();
            formData.append('file', blob, filenameForBlob(blob));
            formData.append('response_format', 'diarized_json');
            formData.append('stream', 'true');
            formData.append('vad', 'false');
            formData.append('chunk_size', '2.0');
            formData.append('chunk_hop', '0.5');

            const response = await fetch('/v1/audio/transcriptions', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            // Incrementally apply events so UI updates in real-time
            const newSegments = [];
            const newEvents = [];
            let newDuration = duration;
            let newSpeakers = speakers;

            let lastUiUpdateAt = 0;
            const maybeUpdateUi = (force = false) => {
                const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
                if (!force && (now - lastUiUpdateAt) < 100) return;
                lastUiUpdateAt = now;
                setSegments([...newSegments]);
                setEvents([...newEvents]);
                setDuration(newDuration);
                setSpeakers(newSpeakers);
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const event = JSON.parse(line);
                            newEvents.push(event);
                            
                            // Process event data locally
                            switch (event.event) {
                                case 'start':
                                    newDuration = event.duration;
                                    newSpeakers = event.speakers || [];
                                    break;
                                case 'segment':
                                    newSegments.push({
                                        start: event.start,
                                        end: event.end,
                                        speaker: event.speaker,
                                        similarity: event.similarity,
                                        vad_confidence: event.vad_confidence
                                    });
                                    break;
                            }

                            maybeUpdateUi(false);
                        } catch (e) {
                            console.error('Error parsing event:', e);
                        }
                    }
                }
            }

            // Flush any remaining buffered line
            if (buffer.trim()) {
                try {
                    const event = JSON.parse(buffer);
                    newEvents.push(event);
                    switch (event.event) {
                        case 'start':
                            newDuration = event.duration;
                            newSpeakers = event.speakers || [];
                            break;
                        case 'segment':
                            newSegments.push({
                                start: event.start,
                                end: event.end,
                                speaker: event.speaker,
                                similarity: event.similarity,
                                vad_confidence: event.vad_confidence
                            });
                            break;
                    }
                } catch (e) {
                    console.error('Error parsing event:', e);
                }
            }

            maybeUpdateUi(true);

            if (!isPartial) {
                setStatus('ready');
                setStatusText('Recording complete - Click play to listen');
            }

        } catch (error) {
            console.error('Error processing diarization:', error);
            if (!isPartial) {
                setStatus('ready');
                setStatusText('Error processing audio');
            }
        } finally {
            processingRef.current = false;
        }
    }, [duration, speakers, filenameForBlob]);

    const processRefinedDiarization = useCallback(async (blob) => {
        try {
            setStatus('processing');
            setStatusText('Running TS-VAD refinement...');

            const formData = new FormData();
            formData.append('file', blob, filenameForBlob(blob));
            formData.append('response_format', 'diarized_json');
            formData.append('stream', 'false');
            formData.append('diarization_mode', 'ts_vad');
            formData.append('window_size', '2.0');
            formData.append('window_hop', '0.5');
            formData.append('vad', 'true');

            const response = await fetch('/v1/audio/transcriptions', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // Update refined segments and unknown speakers
            setRefinedSegments(result.segments || []);
            setUnknownSpeakers(result.unknown_speakers || []);
            setKnownSpeakers(result.known_speakers || []);

            // Add TS-VAD events to event log
            const tsVadEvents = [
                { event: 'ts_vad_start', mode: result.mode, duration: result.duration },
                ...(result.segments || []).map(seg => ({
                    event: 'segment',
                    ...seg
                })),
                { event: 'ts_vad_done', unknown_count: result.unknown_speakers?.length || 0 }
            ];
            setEvents(prev => [...prev, ...tsVadEvents]);

            setStatus('ready');
            setStatusText('TS-VAD refinement complete');

        } catch (error) {
            console.error('Error processing TS-VAD refinement:', error);
            setStatus('ready');
            setStatusText('Error in TS-VAD refinement');
        }
    }, [filenameForBlob]);

    const togglePlayback = useCallback(() => {
        if (playbackWavesurferRef.current) {
            if (playbackWavesurferRef.current.isPlaying()) {
                playbackWavesurferRef.current.pause();
            } else {
                // If there's a selection, play from selection start
                if (selectionRef.current) {
                    const currentTime = playbackWavesurferRef.current.getCurrentTime();
                    // If cursor is outside selection, seek to selection start
                    if (currentTime < selectionRef.current.start || currentTime >= selectionRef.current.end) {
                        const position = selectionRef.current.start / playbackWavesurferRef.current.getDuration();
                        playbackWavesurferRef.current.seekTo(position);
                    }
                }
                playbackWavesurferRef.current.play();
            }
        }
    }, []);

    // Keyboard shortcut for spacebar to toggle playback
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Only trigger if not typing in an input field
            if (e.code === 'Space' && 
                e.target.tagName !== 'INPUT' && 
                e.target.tagName !== 'TEXTAREA' &&
                playbackWavesurferRef.current &&
                audioBlob) {
                e.preventDefault();
                if (playbackWavesurferRef.current.isPlaying()) {
                    playbackWavesurferRef.current.pause();
                } else {
                    // If there's a selection, play from selection start
                    if (selectionRef.current) {
                        const currentTime = playbackWavesurferRef.current.getCurrentTime();
                        // If cursor is outside selection, seek to selection start
                        if (currentTime < selectionRef.current.start || currentTime >= selectionRef.current.end) {
                            const position = selectionRef.current.start / playbackWavesurferRef.current.getDuration();
                            playbackWavesurferRef.current.seekTo(position);
                        }
                    }
                    playbackWavesurferRef.current.play();
                }
            }
        };
        
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [audioBlob]);

    const seekToSegment = useCallback((segment) => {
        if (playbackWavesurferRef.current && duration > 0) {
            // Set active segment for tracking
            setActiveSegment(segment);
            activeSegmentRef.current = segment;
            
            // Clear existing regions and create a new one for this segment
            if (regionsRef.current) {
                regionsRef.current.clearRegions();
                regionsRef.current.addRegion({
                    start: segment.start,
                    end: segment.end,
                    color: 'rgba(0, 210, 255, 0.3)',
                });
                const sel = { start: segment.start, end: segment.end };
                setSelection(sel);
                selectionRef.current = sel;
            }
            
            // Seek to the position (0-1 range)
            const position = segment.start / duration;
            playbackWavesurferRef.current.seekTo(position);
            // Start playing
            playbackWavesurferRef.current.play();
        }
    }, [duration]);

    const clearActiveSegment = useCallback(() => {
        setActiveSegment(null);
        activeSegmentRef.current = null;
    }, []);

    // Fetch known speakers on mount
    useEffect(() => {
        fetch('/v1/speakers')
            .then(res => res.json())
            .then(data => setKnownSpeakers(data.speakers || []))
            .catch(err => console.error('Error fetching speakers:', err));
    }, []);
    
    // Fetch saved recordings on mount
    const fetchRecordings = useCallback(() => {
        fetch('/v1/recordings')
            .then(res => res.json())
            .then(data => setSavedRecordings(data.recordings || []))
            .catch(err => console.error('Error fetching recordings:', err));
    }, []);
    
    useEffect(() => {
        fetchRecordings();
    }, [fetchRecordings]);
    
    // URL routing: load recording from path on mount
    useEffect(() => {
        const path = window.location.pathname;
        const match = path.match(/^\/recording\/([a-zA-Z0-9_-]+)$/);
        if (match) {
            const recordingId = match[1];
            // Fetch recording metadata and load it
            fetch(`/v1/recordings`)
                .then(res => res.json())
                .then(data => {
                    const recording = (data.recordings || []).find(r => r.id === recordingId);
                    if (recording) {
                        loadRecording(recording);
                    } else {
                        console.error('Recording not found:', recordingId);
                        // Navigate to home
                        window.history.replaceState({}, '', '/');
                    }
                })
                .catch(err => console.error('Error loading recording from URL:', err));
        }
    }, []); // Only run once on mount
    
    // Save current recording
    const saveRecording = useCallback(async () => {
        if (!audioBlob || isSavingRecording) return;
        
        setIsSavingRecording(true);
        try {
            const formData = new FormData();
            formData.append('file', audioBlob, filenameForBlob(audioBlob));
            formData.append('name', recordingName.trim() || `Recording ${new Date().toLocaleString()}`);
            formData.append('duration', duration.toString());
            
            const response = await fetch('/v1/recordings', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to save recording');
            }
            
            const result = await response.json();
            console.log('Recording saved:', result);
            
            // Refresh recordings list
            fetchRecordings();
            setRecordingName('');
        } catch (error) {
            console.error('Error saving recording:', error);
            alert('Error saving recording: ' + error.message);
        } finally {
            setIsSavingRecording(false);
        }
    }, [audioBlob, recordingName, duration, fetchRecordings, isSavingRecording, filenameForBlob]);
    
    // Load a saved recording
    const loadRecording = useCallback(async (recording) => {
        if (isLoadingRecording || isRecording) return;
        
        setIsLoadingRecording(true);
        setStatus('processing');
        setStatusText(`Loading "${recording.name}"...`);
        setCurrentRecordingId(recording.id);
        
        // Update URL without page reload
        window.history.pushState({ recordingId: recording.id }, '', `/recording/${recording.id}`);
        
        try {
            // Fetch the audio file
            const response = await fetch(`/v1/recordings/${recording.id}`);
            if (!response.ok) {
                throw new Error('Failed to fetch recording');
            }
            
            const blob = await response.blob();
            
            // Clear previous state
            setSegments([]);
            setRefinedSegments([]);
            setUnknownSpeakers([]);
            setEvents([]);
            setSpeakers([]);
            speakerMap.clear();
            setActiveSegment(null);
            activeSegmentRef.current = null;
            
            // Cleanup previous playback wavesurfer
            if (playbackWavesurferRef.current) {
                playbackWavesurferRef.current.destroy();
                playbackWavesurferRef.current = null;
            }
            
            // Set the audio blob (this will trigger playback waveform creation)
            setAudioBlob(blob);
            
            // Process diarization (coarse streaming)
            await processStreamingDiarization(blob, false);
            
            // Process TS-VAD refinement (optional - uncomment to enable)
            await processRefinedDiarization(blob);
            
        } catch (error) {
            console.error('Error loading recording:', error);
            setStatus('ready');
            setStatusText('Error loading recording');
        } finally {
            setIsLoadingRecording(false);
        }
    }, [isLoadingRecording, isRecording, processStreamingDiarization, processRefinedDiarization, speakerMap]);
    
    // Delete a saved recording
    const deleteRecording = useCallback(async (recording, e) => {
        e.stopPropagation();
        
        if (!confirm(`Delete "${recording.name}"?`)) return;
        
        try {
            const response = await fetch(`/v1/recordings/${recording.id}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to delete recording');
            }
            
            // Refresh recordings list
            fetchRecordings();
        } catch (error) {
            console.error('Error deleting recording:', error);
            alert('Error deleting recording: ' + error.message);
        }
    }, [fetchRecordings]);
    
    // Format file size
    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };
    
    // Format date
    const formatDate = (isoString) => {
        const date = new Date(isoString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    const openNamingModal = useCallback((segment, e) => {
        e.stopPropagation();
        setNamingSegment(segment);
        setNewSpeakerName('');
        setSelectedExistingSpeaker(null);
        // Refresh speaker list
        fetch('/v1/speakers')
            .then(res => res.json())
            .then(data => setKnownSpeakers(data.speakers || []))
            .catch(err => console.error('Error fetching speakers:', err));
    }, []);

    const closeNamingModal = useCallback(() => {
        setNamingSegment(null);
        setNewSpeakerName('');
        setSelectedExistingSpeaker(null);
    }, []);

    const saveSegmentSpeaker = useCallback(async () => {
        if (!namingSegment || !audioBlob) return;
        
        const speakerName = selectedExistingSpeaker || newSpeakerName.trim();
        if (!speakerName) return;
        
        setIsSaving(true);
        try {
            const formData = new FormData();
            formData.append('file', audioBlob, filenameForBlob(audioBlob));
            formData.append('name', speakerName);
            formData.append('start', namingSegment.start.toString());
            formData.append('end', namingSegment.end.toString());
            
            const response = await fetch('/v1/speakers/from-segment', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to save speaker');
            }
            
            const result = await response.json();
            console.log('Speaker saved:', result);
            
            // Update ALL segments with the same speaker cluster label
            const oldSpeakerLabel = namingSegment.speaker;
            setSegments(prev => prev.map(seg => 
                seg.speaker === oldSpeakerLabel
                    ? { ...seg, speaker: speakerName }
                    : seg
            ));
            setRefinedSegments(prev => prev.map(seg => 
                seg.speaker === oldSpeakerLabel
                    ? { ...seg, speaker: speakerName }
                    : seg
            ));
            
            // Remove from unknown speakers if it was an unknown cluster
            if (oldSpeakerLabel && oldSpeakerLabel.startsWith('unknown_')) {
                setUnknownSpeakers(prev => prev.filter(u => u !== oldSpeakerLabel));
            }
            
            // Update known speakers list
            if (!knownSpeakers.includes(speakerName)) {
                setKnownSpeakers(prev => [...prev, speakerName]);
            }
            
            // Update speaker color map
            getSpeakerColor(speakerName, speakerMap);
            
            closeNamingModal();
        } catch (error) {
            console.error('Error saving speaker:', error);
            alert('Error saving speaker: ' + error.message);
        } finally {
            setIsSaving(false);
        }
    }, [namingSegment, audioBlob, selectedExistingSpeaker, newSpeakerName, knownSpeakers, speakerMap, closeNamingModal, filenameForBlob]);

    const reprocessAudio = useCallback(async () => {
        if (!audioBlob || processingRef.current) return;
        
        setStatus('processing');
        setStatusText('Reprocessing with updated speaker database...');
        
        // Clear segments to show fresh results
        await processStreamingDiarization(audioBlob, false);
        
        // Also re-run TS-VAD refinement (optional)
        await processRefinedDiarization(audioBlob);
    }, [audioBlob, processStreamingDiarization, processRefinedDiarization]);

    // Compare all segments against a specific speaker
    const compareToSpeaker = useCallback(async (speakerName) => {
        if (!audioBlob || !segments.length) return;
        
        if (compareSpeaker === speakerName) {
            // Toggle off if clicking the same speaker
            setCompareSpeaker(null);
            setSegmentSimilarities({});
            return;
        }
        
        setCompareSpeaker(speakerName);
        setSegmentSimilarities({});
        
        try {
            const formData = new FormData();
            formData.append('file', audioBlob, filenameForBlob(audioBlob));
            formData.append('segments', JSON.stringify(segments.map(s => ({ start: s.start, end: s.end }))));
            
            const response = await fetch(`/v1/speakers/${encodeURIComponent(speakerName)}/compare`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to compare segments');
            }
            
            const result = await response.json();
            const similarities = {};
            for (const seg of result.segments) {
                const key = `${seg.start}-${seg.end}`;
                similarities[key] = seg.similarity;
            }
            setSegmentSimilarities(similarities);
        } catch (error) {
            console.error('Error comparing to speaker:', error);
            setCompareSpeaker(null);
        }
    }, [audioBlob, segments, compareSpeaker, filenameForBlob]);

    // Identify a single segment against speaker database
    const identifySegment = useCallback(async (seg, e) => {
        e.stopPropagation();
        if (!audioBlob) return;
        
        const segKey = `${seg.start}-${seg.end}`;
        setIdentifyingSegment(segKey);
        
        try {
            const formData = new FormData();
            formData.append('file', audioBlob, filenameForBlob(audioBlob));
            formData.append('start', seg.start.toString());
            formData.append('end', seg.end.toString());
            
            const response = await fetch('/v1/speakers/identify-segment', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to identify segment');
            }
            
            const result = await response.json();
            setSegmentIdentifications(prev => ({
                ...prev,
                [segKey]: result
            }));
            
            // If we got a top match, update the segment's speaker
            if (result.top_match && result.top_match.similarity >= 0.5) {
                setSegments(prev => prev.map(s => 
                    s.start === seg.start && s.end === seg.end
                        ? { ...s, speaker: result.top_match.name }
                        : s
                ));
                // Update speaker color
                getSpeakerColor(result.top_match.name, speakerMap);
            }
        } catch (error) {
            console.error('Error identifying segment:', error);
        } finally {
            setIdentifyingSegment(null);
        }
    }, [audioBlob, speakerMap, filenameForBlob]);

    return (
        <div className="container">
            <h1>🎤 VectorMe Voice Recorder</h1>
            
            <div className="card">
                <div className={`status ${status}`}>
                    {status === 'recording' && (
                        <span className="live-indicator">
                            <span className="live-dot"></span>
                        </span>
                    )}
                    {statusText}
                </div>

                <div className="waveform-container" ref={waveformRef}>
                    {isRecording ? (
                        <canvas 
                            ref={canvasRef} 
                            width={800} 
                            height={50}
                            style={{ width: '100%', height: '50px', borderRadius: '8px', position: 'relative', zIndex: 2 }}
                        />
                    ) : !audioBlob ? (
                        <div style={{ 
                            height: '50px', 
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'center',
                            color: '#666'
                        }}>
                            Click Record to start
                        </div>
                    ) : null}
                </div>

                <div className="controls">
                    {!isRecording ? (
                        <button 
                            className="btn-record" 
                            onClick={startRecording}
                            disabled={status === 'processing'}
                        >
                            🎙️ Record
                        </button>
                    ) : (
                        <>
                            <button 
                                className="btn-play" 
                                onClick={isPaused ? resumeRecording : pauseRecording}
                                style={{ background: isPaused ? 'linear-gradient(135deg, #11998e, #38ef7d)' : 'linear-gradient(135deg, #f7971e, #ffd200)', color: isPaused ? 'white' : '#1a1a2e' }}
                            >
                                {isPaused ? '▶️ Resume' : '⏸️ Pause'}
                            </button>
                            <button 
                                className="btn-stop recording" 
                                onClick={stopRecording}
                            >
                                ⏹️ Stop
                            </button>
                        </>
                    )}
                </div>
                
                {/* Live diarization display during recording */}
                {isRecording && segments.length > 0 && (
                    <div style={{ marginTop: '16px' }}>
                        <div style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '8px',
                            marginBottom: '12px',
                            color: '#4a9eff'
                        }}>
                            <span className="live-dot" style={{ width: '8px', height: '8px', background: '#ff416c', borderRadius: '50%', animation: 'pulse 1.5s infinite' }}></span>
                            <span style={{ fontWeight: 600 }}>Live Speaker Detection</span>
                            <span style={{ color: '#888', fontSize: '13px' }}>({segments.length} segments)</span>
                        </div>
                        <div className="diarization-results" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                            {segments.slice().reverse().map((seg, i) => {
                                const isUnknownSpeaker = seg.speaker && seg.speaker.startsWith('unknown_');
                                const displaySpeaker = isUnknownSpeaker 
                                    ? `Unknown (${seg.speaker})`
                                    : (seg.speaker || 'Unknown Speaker');
                                const similarityPercent = seg.similarity ? Math.round(seg.similarity * 100) : null;
                                const similarityColor = seg.similarity >= 0.7 ? '#38ef7d' : seg.similarity >= 0.5 ? '#ffd200' : '#ff6b8a';
                                return (
                                    <div 
                                        key={i} 
                                        className="segment"
                                        style={{ 
                                            opacity: i === 0 ? 1 : 0.7,
                                            transition: 'opacity 0.3s'
                                        }}
                                    >
                                        <span className="segment-time">
                                            {formatTime(seg.start)} → {formatTime(seg.end)}
                                        </span>
                                        <span 
                                            className={`segment-speaker ${seg.speaker && !isUnknownSpeaker ? 'known' : 'unknown'}`}
                                            style={seg.speaker && !isUnknownSpeaker ? { background: getSpeakerColor(seg.speaker, speakerMap) } : {}}
                                        >
                                            {displaySpeaker}
                                        </span>
                                        {similarityPercent !== null && (
                                            <span 
                                                style={{
                                                    color: similarityColor,
                                                    fontWeight: 600,
                                                    fontSize: '12px',
                                                    marginLeft: '8px',
                                                    fontFamily: "'Monaco', 'Consolas', monospace"
                                                }}
                                                title={`Speaker match confidence: ${similarityPercent}%`}
                                            >
                                                {similarityPercent}%
                                            </span>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>

            {audioBlob && (
                <div className="card">
                    <h3>📼 Recorded Audio</h3>
                    <div className="waveform-container" ref={playbackWaveformRef}></div>
                    <div className="playback-time" style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '4px 8px',
                        fontFamily: "'Monaco', 'Consolas', monospace",
                        fontSize: '13px',
                        color: '#4a9eff',
                        marginBottom: '8px'
                    }}>
                        <span>{formatTime(currentTime)}</span>
                        <span style={{ color: '#888' }}>{selection ? `Selection: ${formatTime(selection.start)} - ${formatTime(selection.end)}` : 'Drag to select • Space to play/pause'}</span>
                        <span>{formatTime(duration)}</span>
                    </div>
                    <div className="controls">
                        <button 
                            className="btn-play" 
                            onClick={togglePlayback}
                            disabled={!playbackWavesurferRef.current}
                        >
                            {isPlaying ? '⏸️ Pause' : '▶️ Play'}
                        </button>
                        {selection && (
                            <button 
                                className="btn-play" 
                                onClick={() => {
                                    const regions = regionsRef.current?.getRegions();
                                    if (regions && regions.length > 0) {
                                        regions[0].play();
                                    }
                                }}
                                style={{ background: 'linear-gradient(135deg, #667eea, #764ba2)' }}
                            >
                                ▶️ Play Selection
                            </button>
                        )}
                        {selection && (
                            <button 
                                className="btn-play" 
                                onClick={() => {
                                    regionsRef.current?.clearRegions();
                                    setSelection(null);
                                    selectionRef.current = null;
                                }}
                                style={{ background: 'rgba(255, 255, 255, 0.1)', color: '#888' }}
                            >
                                ✕ Clear
                            </button>
                        )}
                        {selection && (
                            <button 
                                className="btn-play" 
                                onClick={(e) => openNamingModal({ start: selection.start, end: selection.end }, e)}
                                style={{ background: 'linear-gradient(135deg, #00d2ff, #3a7bd5)' }}
                            >
                                🏷️ Name Speaker
                            </button>
                        )}
                        {selection && (
                            <button 
                                className="btn-play" 
                                onClick={(e) => identifySegment({ start: selection.start, end: selection.end }, e)}
                                disabled={identifyingSegment === `${selection.start}-${selection.end}`}
                                style={{ background: 'linear-gradient(135deg, #f7971e, #ffd200)', color: '#1a1a2e' }}
                            >
                                {identifyingSegment === `${selection.start}-${selection.end}` ? '⏳ Identifying...' : '🔍 Identify Speaker'}
                            </button>
                        )}
                    </div>
                    
                    {/* Show identification results for selection */}
                    {selection && segmentIdentifications[`${selection.start}-${selection.end}`] && (
                        <div style={{
                            marginTop: '12px',
                            padding: '12px 16px',
                            background: 'rgba(0, 0, 0, 0.3)',
                            borderRadius: '8px',
                            fontSize: '14px'
                        }}>
                            <div style={{ marginBottom: '8px', color: '#a0b0ff', fontWeight: 600 }}>
                                🔍 Identification Results
                            </div>
                            {segmentIdentifications[`${selection.start}-${selection.end}`].matches?.length > 0 ? (
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                    {segmentIdentifications[`${selection.start}-${selection.end}`].matches.map((match, idx) => (
                                        <span 
                                            key={match.name}
                                            style={{
                                                padding: '6px 12px',
                                                borderRadius: '16px',
                                                background: idx === 0 ? getSpeakerColor(match.name, speakerMap) : 'rgba(255, 255, 255, 0.1)',
                                                color: idx === 0 ? 'white' : '#888',
                                                fontFamily: "'Monaco', 'Consolas', monospace",
                                                fontSize: '13px',
                                                border: idx === 0 ? '2px solid rgba(255, 255, 255, 0.5)' : 'none'
                                            }}
                                        >
                                            {match.name}: {(match.similarity * 100).toFixed(0)}%
                                        </span>
                                    ))}
                                </div>
                            ) : (
                                <span style={{ color: '#888' }}>No matching speakers found</span>
                            )}
                        </div>
                    )}
                    
                    {/* Save Recording Form */}
                    <div className="save-recording-form" style={{ marginTop: '16px' }}>
                        <input
                            type="text"
                            placeholder="Name this recording..."
                            value={recordingName}
                            onChange={(e) => setRecordingName(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') saveRecording();
                            }}
                            aria-label="Recording name"
                        />
                        <button
                            className="btn-save"
                            onClick={saveRecording}
                            disabled={isSavingRecording}
                            title="Save recording for later analysis"
                        >
                            {isSavingRecording ? '💾 Saving...' : '💾 Save'}
                        </button>
                        {currentRecordingId && (
                            <button
                                className="btn-share"
                                onClick={() => {
                                    const url = `${window.location.origin}/recording/${currentRecordingId}`;
                                    navigator.clipboard.writeText(url).then(() => {
                                        // Show brief feedback
                                        const btn = event.target;
                                        const originalText = btn.innerText;
                                        btn.innerText = '✓ Copied!';
                                        setTimeout(() => btn.innerText = originalText, 2000);
                                    }).catch(err => {
                                        console.error('Failed to copy:', err);
                                        prompt('Copy this link:', url);
                                    });
                                }}
                                title="Copy link to this recording"
                            >
                                🔗 Copy Link
                            </button>
                        )}
                    </div>
                </div>
            )}

            {audioBlob && (
                <div className="card">
                    <div className="card-header">
                        <h3>🎯 Speaker Diarization</h3>
                        <button 
                            className="btn-reprocess" 
                            onClick={reprocessAudio}
                            disabled={status === 'processing' || isRecording}
                            title="Re-analyze audio with updated speaker database"
                        >
                            🔄 Reprocess
                        </button>
                    </div>
                    
                    {speakers.length > 0 && (
                        <div className="speaker-list">
                            <span style={{ color: '#888', marginRight: '8px' }}>Compare to speaker:</span>
                            {speakers.map((speaker, i) => (
                                <span 
                                    key={speaker} 
                                    className="speaker-badge"
                                    onClick={() => compareToSpeaker(speaker)}
                                    style={{ 
                                        background: getSpeakerColor(speaker, speakerMap),
                                        cursor: 'pointer',
                                        border: compareSpeaker === speaker ? '2px solid #fff' : '2px solid transparent',
                                        boxShadow: compareSpeaker === speaker ? '0 0 12px rgba(255, 255, 255, 0.5)' : undefined
                                    }}
                                    title={compareSpeaker === speaker ? 'Click to clear comparison' : `Click to see similarity of all segments to ${speaker}`}
                                >
                                    {speaker}
                                </span>
                            ))}
                            {compareSpeaker && (
                                <button
                                    onClick={() => { setCompareSpeaker(null); setSegmentSimilarities({}); }}
                                    style={{
                                        background: 'rgba(255, 255, 255, 0.1)',
                                        border: 'none',
                                        color: '#888',
                                        padding: '4px 8px',
                                        borderRadius: '4px',
                                        cursor: 'pointer',
                                        fontSize: '12px'
                                    }}
                                >
                                    ✕ Clear
                                </button>
                            )}
                        </div>
                    )}

                    {duration > 0 && (
                        <>
                        <div 
                            className="timeline"
                            onDoubleClick={() => {
                                if (selection) {
                                    regionsRef.current?.clearRegions();
                                    setSelection(null);
                                    selectionRef.current = null;
                                }
                            }}
                        >
                            {(() => {
                                // Show only refined segments if available, otherwise show streaming segments
                                const allSegments = (refinedSegments && refinedSegments.length > 0) 
                                    ? refinedSegments 
                                    : (segments || []);
                                return allSegments.sort((a, b) => a.start - b.start).map((seg, i) => {
                                const left = (seg.start / duration) * 100;
                                const width = ((seg.end - seg.start) / duration) * 100;
                                const isActive = activeSegment && activeSegment.start === seg.start && activeSegment.end === seg.end;
                                const isUnknownSpeaker = seg.speaker && seg.speaker.startsWith('unknown_');
                                const displaySpeaker = isUnknownSpeaker 
                                    ? seg.speaker.replace('unknown_', 'U')
                                    : (seg.speaker || '?');
                                return (
                                    <div
                                        key={i}
                                        className={`timeline-segment ${isActive ? 'active' : ''}`}
                                        onClick={() => seekToSegment(seg)}
                                        style={{
                                            left: `${left}%`,
                                            width: `${width}%`,
                                            background: (seg.speaker && !isUnknownSpeaker) ? getSpeakerColor(seg.speaker, speakerMap) : '#666',
                                            cursor: audioBlob ? 'pointer' : 'default'
                                        }}
                                        title={`${seg.speaker || 'Unknown'}: ${formatTime(seg.start)} - ${formatTime(seg.end)}${audioBlob ? ' (click to play)' : ''}`}
                                    >
                                        {width > 10 ? displaySpeaker : ''}
                                    </div>
                                );
                                });
                            })()}
                        </div>
                        {activeSegment && (
                            <div className="timeline-info">
                                <div className="timeline-info-item">
                                    <span className="timeline-info-label">Speaker:</span>
                                    <span className="timeline-info-value" style={{ color: getSpeakerColor(activeSegment.speaker, speakerMap) }}>
                                        {activeSegment.speaker || 'Unknown'}
                                    </span>
                                </div>
                                <div className="timeline-info-item">
                                    <span className="timeline-info-label">Start:</span>
                                    <span className="timeline-info-value">{formatTime(activeSegment.start)}</span>
                                </div>
                                <div className="timeline-info-item">
                                    <span className="timeline-info-label">Duration:</span>
                                    <span className="timeline-info-value">{(activeSegment.end - activeSegment.start).toFixed(1)}s</span>
                                </div>
                                <div className="timeline-info-item">
                                    <span className="timeline-info-label">End:</span>
                                    <span className="timeline-info-value">{formatTime(activeSegment.end)}</span>
                                </div>
                                <button 
                                    onClick={clearActiveSegment}
                                    style={{ 
                                        background: 'transparent', 
                                        border: 'none', 
                                        color: '#888', 
                                        cursor: 'pointer',
                                        padding: '0 4px',
                                        fontSize: '16px'
                                    }}
                                    title="Clear selection"
                                >✕</button>
                            </div>
                        )}
                        </>
                    )}

                    <div className="diarization-results">
                        {(() => {
                            // Show only refined segments if available, otherwise show streaming segments
                            const allSegments = (refinedSegments && refinedSegments.length > 0) 
                                ? refinedSegments 
                                : (segments || []);
                            return allSegments.sort((a, b) => a.start - b.start).map((seg, i) => {
                            const isActive = activeSegment && activeSegment.start === seg.start && activeSegment.end === seg.end;
                            const segKey = `${seg.start}-${seg.end}`;
                            const similarity = segmentSimilarities[segKey];
                            const isUnknownSpeaker = seg.speaker && seg.speaker.startsWith('unknown_');
                            const displaySpeaker = isUnknownSpeaker 
                                ? `Unknown (${seg.speaker})`
                                : (seg.speaker || 'Unknown Speaker');
                            
                            return (
                            <div 
                                key={i} 
                                className={`segment ${isActive ? 'active' : ''}`}
                                onClick={() => seekToSegment(seg)}
                                style={{ cursor: audioBlob ? 'pointer' : 'default' }}
                                title={audioBlob ? 'Click to play from this segment' : 'Record audio to enable playback'}
                            >
                                <span className="segment-time">
                                    {formatTime(seg.start)} → {formatTime(seg.end)}
                                </span>
                                {seg.cause && (
                                    <span 
                                        style={{
                                            fontSize: '10px',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            background: seg.cause === 'ts_vad' ? 'rgba(56, 239, 125, 0.2)' : 'rgba(74, 158, 255, 0.2)',
                                            color: seg.cause === 'ts_vad' ? '#38ef7d' : '#4a9eff',
                                            fontWeight: 600,
                                            marginLeft: '4px'
                                        }}
                                        title={seg.cause === 'ts_vad' ? 'TS-VAD refined segment' : 'Streaming coarse segment'}
                                    >
                                        {seg.cause === 'ts_vad' ? '🎯 TS-VAD' : '🌊 Stream'}
                                    </span>
                                )}
                                {seg.vad_confidence !== undefined && (
                                    <span 
                                        className={`segment-vad-confidence ${seg.vad_confidence >= 0.8 ? 'high' : seg.vad_confidence >= 0.5 ? 'medium' : 'low'}`}
                                        title={`VAD confidence: ${(seg.vad_confidence * 100).toFixed(1)}%`}
                                    >
                                        🎤 {(seg.vad_confidence * 100).toFixed(0)}%
                                    </span>
                                )}
                                <span 
                                    className={`segment-speaker ${seg.speaker && !isUnknownSpeaker ? 'known' : 'unknown'}`}
                                    style={seg.speaker && !isUnknownSpeaker ? { background: getSpeakerColor(seg.speaker, speakerMap) } : {}}
                                >
                                    {displaySpeaker}
                                </span>
                                {/* Show segment's intrinsic similarity (from backend) when not comparing */}
                                {!compareSpeaker && seg.similarity !== undefined && seg.similarity !== null && (
                                    <span 
                                        className="segment-similarity"
                                        style={{
                                            color: seg.similarity >= 0.7 ? '#38ef7d' : seg.similarity >= 0.5 ? '#ffd200' : '#ff6b8a',
                                            fontWeight: 600,
                                            fontSize: '13px',
                                            marginLeft: '8px',
                                            fontFamily: "'Monaco', 'Consolas', monospace"
                                        }}
                                        title={`Speaker match confidence: ${(seg.similarity * 100).toFixed(1)}%`}
                                    >
                                        {(seg.similarity * 100).toFixed(0)}%
                                    </span>
                                )}
                                {/* Show comparison similarity when comparing to a specific speaker */}
                                {compareSpeaker && similarity !== undefined && (
                                    <span 
                                        className="segment-similarity"
                                        style={{
                                            color: similarity >= 0.7 ? '#38ef7d' : similarity >= 0.5 ? '#ffd200' : '#ff6b8a',
                                            fontWeight: 600,
                                            fontSize: '13px',
                                            marginLeft: '8px',
                                            fontFamily: "'Monaco', 'Consolas', monospace"
                                        }}
                                        title={`Similarity to ${compareSpeaker}: ${(similarity * 100).toFixed(1)}%`}
                                    >
                                        {similarity !== null ? `${(similarity * 100).toFixed(0)}%` : '—'}
                                    </span>
                                )}
                                {audioBlob && <span className="segment-play-icon">{isActive ? '⏸' : '▶'}</span>}
                                {audioBlob && (
                                    <button 
                                        className="segment-name-btn"
                                        onClick={(e) => identifySegment(seg, e)}
                                        disabled={identifyingSegment === segKey}
                                        title="Identify speaker for this segment"
                                        style={{ marginLeft: '4px' }}
                                    >
                                        {identifyingSegment === segKey ? '⏳' : '🔍'} ID
                                    </button>
                                )}
                                {audioBlob && (
                                    <button 
                                        className="segment-name-btn"
                                        onClick={(e) => openNamingModal(seg, e)}
                                        title="Name this speaker"
                                    >
                                        ✏️ Name
                                    </button>
                                )}
                                {segmentIdentifications[segKey]?.top_match && (
                                    <span 
                                        style={{
                                            fontSize: '11px',
                                            color: segmentIdentifications[segKey].top_match.similarity >= 0.7 ? '#38ef7d' : 
                                                   segmentIdentifications[segKey].top_match.similarity >= 0.5 ? '#ffd200' : '#ff6b8a',
                                            marginLeft: '8px',
                                            fontFamily: "'Monaco', 'Consolas', monospace"
                                        }}
                                        title={`Top matches: ${segmentIdentifications[segKey].matches.map(m => `${m.name}: ${(m.similarity * 100).toFixed(0)}%`).join(', ')}`}
                                    >
                                        → {segmentIdentifications[segKey].top_match.name} ({(segmentIdentifications[segKey].top_match.similarity * 100).toFixed(0)}%)
                                    </span>
                                )}
                            </div>
                            );
                            });
                        })()}
                    </div>
                </div>
            )}

            {namingSegment && (
                <div className="modal-overlay" onClick={closeNamingModal}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="modal-title">🏷️ Name This Speaker</div>
                        
                        <div className="modal-info" style={{ marginBottom: '16px' }}>
                            Segment: {formatTime(namingSegment.start)} → {formatTime(namingSegment.end)}
                            {namingSegment.speaker && ` (currently: ${namingSegment.speaker})`}
                        </div>

                        {knownSpeakers.length > 0 && (
                            <div className="modal-section">
                                <div className="modal-section-title">Add to existing speaker</div>
                                <div className="modal-speakers">
                                    {knownSpeakers.map(speaker => (
                                        <div
                                            key={speaker}
                                            className={`modal-speaker-option ${selectedExistingSpeaker === speaker ? 'selected' : ''}`}
                                            style={{ background: getSpeakerColor(speaker, speakerMap) }}
                                            onClick={() => {
                                                setSelectedExistingSpeaker(speaker);
                                                setNewSpeakerName('');
                                            }}
                                        >
                                            {speaker}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        <div className="modal-section">
                            <div className="modal-section-title">
                                {knownSpeakers.length > 0 ? 'Or create new speaker' : 'Create new speaker'}
                            </div>
                            <input
                                type="text"
                                className="modal-input"
                                placeholder="Enter speaker name..."
                                value={newSpeakerName}
                                onChange={(e) => {
                                    setNewSpeakerName(e.target.value);
                                    setSelectedExistingSpeaker(null);
                                }}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && (newSpeakerName.trim() || selectedExistingSpeaker)) {
                                        saveSegmentSpeaker();
                                    }
                                }}
                                autoFocus
                            />
                        </div>

                        <div className="modal-buttons">
                            <button className="modal-btn modal-btn-cancel" onClick={closeNamingModal}>
                                Cancel
                            </button>
                            <button 
                                className="modal-btn modal-btn-save" 
                                onClick={saveSegmentSpeaker}
                                disabled={isSaving || (!newSpeakerName.trim() && !selectedExistingSpeaker)}
                            >
                                {isSaving ? 'Saving...' : 'Save Speaker'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {events.length > 0 && (
                <div className="card">
                    <h3>📋 Events Log</h3>
                    <div className="events-log">
                        {events.map((evt, i) => (
                            <div key={i} className="event-item">
                                <span className="event-type">[{evt.event}]</span>{' '}
                                {JSON.stringify(evt)}
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {/* Saved Recordings Section */}
            <div className="card">
                <div className="card-title-row">
                    <h3>📁 Saved Recordings</h3>
                    <button 
                        className="btn-refresh" 
                        onClick={fetchRecordings}
                        title="Refresh recordings list"
                        aria-label="Refresh recordings list"
                    >
                        🔄
                    </button>
                </div>
                
                {savedRecordings.length === 0 ? (
                    <div className="empty-state">
                        No saved recordings yet. Record audio and click Save to keep it for later analysis.
                    </div>
                ) : (
                    <div className="recordings-list">
                        {savedRecordings.map((recording) => (
                            <div 
                                key={recording.id} 
                                className="recording-item"
                                onClick={() => loadRecording(recording)}
                                style={{ cursor: isLoadingRecording ? 'wait' : 'pointer' }}
                                role="button"
                                tabIndex={0}
                                aria-label={`Load recording: ${recording.name}`}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' || e.key === ' ') {
                                        e.preventDefault();
                                        loadRecording(recording);
                                    }
                                }}
                            >
                                <span className="recording-icon">🎵</span>
                                <div className="recording-info">
                                    <div className="recording-name">{recording.name}</div>
                                    <div className="recording-meta">
                                        <span>{formatDate(recording.timestamp)}</span>
                                        <span>{recording.duration > 0 ? formatTime(recording.duration) : '--'}</span>
                                        <span>{formatSize(recording.size)}</span>
                                    </div>
                                </div>
                                <div className="recording-actions">
                                    <button
                                        className="btn-small btn-load"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            loadRecording(recording);
                                        }}
                                        disabled={isLoadingRecording || isRecording}
                                        aria-label={`Load ${recording.name}`}
                                    >
                                        ▶️ Load
                                    </button>
                                    <button
                                        className="btn-small btn-delete"
                                        onClick={(e) => deleteRecording(recording, e)}
                                        aria-label={`Delete ${recording.name}`}
                                    >
                                        🗑️
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<VoiceRecorder />);
