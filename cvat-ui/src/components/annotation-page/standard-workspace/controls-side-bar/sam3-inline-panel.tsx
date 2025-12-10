// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * SAM3 Inline Panel
 *
 * Unified UI for SAM3 capabilities in CVAT:
 * - Text-to-Segment: Segment objects using text prompt
 * - Refine with Clicks: Interactive refinement
 * - Text-to-Detect: Detect all instances matching text
 * - Text-to-Track: Video tracking with text/mask initialization
 */

import React, {
    useState,
    useEffect,
    useCallback,
    useMemo,
} from 'react';
import {
    Button,
    Input,
    Radio,
    Slider,
    Space,
    Typography,
    Spin,
    Tooltip,
    Badge,
    List,
    Modal,
    Empty,
    Alert,
    Divider,
    Switch,
    Card,
} from 'antd';
import {
    SearchOutlined,
    ThunderboltOutlined,
    CheckCircleOutlined,
    ClearOutlined,
    EyeOutlined,
    PlayCircleOutlined,
    StopOutlined,
    AimOutlined,
    ScanOutlined,
    VideoCameraOutlined,
} from '@ant-design/icons';

import { Job, Label, ObjectState, ObjectType, ShapeType } from 'cvat-core-wrapper';
import { getCore } from 'cvat-core-wrapper';

const { Text, Title } = Typography;
const { TextArea } = Input;
const core = getCore();

// API endpoints
const API_BASE = '/api/lambda/sam3';

// Helper to get CSRF token from cookie
function getCsrfToken(): string {
    const name = 'csrftoken';
    const cookies = document.cookie.split(';');
    for (const cookie of cookies) {
        const [cookieName, cookieValue] = cookie.trim().split('=');
        if (cookieName === name) {
            return decodeURIComponent(cookieValue);
        }
    }
    return '';
}

// Helper for authenticated fetch
async function authenticatedFetch(url: string, options: RequestInit = {}): Promise<Response> {
    const defaultHeaders: Record<string, string> = {
        'Content-Type': 'application/json',
    };

    if (options.method && options.method !== 'GET') {
        defaultHeaders['X-CSRFToken'] = getCsrfToken();
    }

    return fetch(url, {
        ...options,
        headers: {
            ...defaultHeaders,
            ...(options.headers || {}),
        },
        credentials: 'same-origin',
    });
}

// Types
type SAM3Mode = 'segment' | 'detect' | 'track';

interface Detection {
    label: string;
    bbox: number[];
    polygon: number[];
    mask?: number[][];
    score: number;
    selected?: boolean;
}

interface SegmentResult {
    mask: number[][];
    bounds: number[];
    points: number[][];
}

interface TrackingState {
    sessionId: string | null;
    isTracking: boolean;
    frameIdx: number;
    textPrompt: string | null;
}

interface Props {
    jobInstance: Job;
    frame: number;
    labels: Label[];
    curZOrder: number;
    createAnnotations: (states: ObjectState[]) => Promise<void>;
}

function SAM3InlinePanel(props: Props): JSX.Element {
    const {
        jobInstance,
        frame,
        labels,
        curZOrder,
        createAnnotations,
    } = props;

    // State
    const [loading, setLoading] = useState(false);
    const [mode, setMode] = useState<SAM3Mode>('segment');
    const [textPrompt, setTextPrompt] = useState('');
    const [threshold, setThreshold] = useState(0.25);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // Results
    const [detections, setDetections] = useState<Detection[]>([]);
    const [segmentResult, setSegmentResult] = useState<SegmentResult | null>(null);

    // Tracking state
    const [tracking, setTracking] = useState<TrackingState>({
        sessionId: null,
        isTracking: false,
        frameIdx: -1,
        textPrompt: null,
    });

    // Output type
    const [outputType, setOutputType] = useState<'polygon' | 'rectangle'>('polygon');

    // Clear messages after timeout
    useEffect(() => {
        if (success) {
            const timer = setTimeout(() => setSuccess(null), 3000);
            return () => clearTimeout(timer);
        }
        return undefined;
    }, [success]);

    useEffect(() => {
        if (error) {
            const timer = setTimeout(() => setError(null), 5000);
            return () => clearTimeout(timer);
        }
        return undefined;
    }, [error]);

    // ========== Text-to-Segment ==========
    const handleSegmentText = useCallback(async () => {
        if (!textPrompt.trim()) {
            setError('Please enter a text prompt');
            return;
        }

        setLoading(true);
        setError(null);
        setSegmentResult(null);

        try {
            const response = await authenticatedFetch(`${API_BASE}/segment-text`, {
                method: 'POST',
                body: JSON.stringify({
                    job_id: jobInstance.id,
                    frame,
                    text: textPrompt.trim(),
                }),
            });

            const data = await response.json();
            if (response.ok) {
                if (data.masks && data.masks.length > 0) {
                    // For now, show first result
                    setSegmentResult({
                        mask: data.masks[0],
                        bounds: data.boxes?.[0] || [],
                        points: [], // Will be computed
                    });
                    setSuccess(`Segmented "${textPrompt}" successfully`);
                } else {
                    setError(`No objects found matching "${textPrompt}"`);
                }
            } else {
                setError(data.error || 'Segmentation failed');
            }
        } catch (err) {
            setError('Failed to connect to SAM3 service');
        } finally {
            setLoading(false);
        }
    }, [textPrompt, jobInstance.id, frame]);

    // ========== Text-to-Detect ==========
    const handleDetect = useCallback(async () => {
        if (!textPrompt.trim()) {
            setError('Please enter a text prompt');
            return;
        }

        setLoading(true);
        setError(null);
        setDetections([]);

        try {
            const response = await authenticatedFetch(`${API_BASE}/detect`, {
                method: 'POST',
                body: JSON.stringify({
                    job_id: jobInstance.id,
                    frame,
                    text: textPrompt.trim(),
                    threshold,
                }),
            });

            const data = await response.json();
            if (response.ok) {
                if (data.detections && data.detections.length > 0) {
                    // Add selected flag to all detections
                    const dets = data.detections.map((d: Detection) => ({
                        ...d,
                        selected: true,
                    }));
                    setDetections(dets);
                    setSuccess(`Found ${dets.length} "${textPrompt}" instances`);
                } else {
                    setError(`No "${textPrompt}" found with confidence >= ${threshold}`);
                }
            } else {
                setError(data.error || 'Detection failed');
            }
        } catch (err) {
            setError('Failed to connect to SAM3 service');
        } finally {
            setLoading(false);
        }
    }, [textPrompt, jobInstance.id, frame, threshold]);

    // ========== Text-to-Track: Init ==========
    const handleTrackInit = useCallback(async () => {
        if (!textPrompt.trim()) {
            setError('Please enter a text prompt to track');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await authenticatedFetch(`${API_BASE}/track/init`, {
                method: 'POST',
                body: JSON.stringify({
                    job_id: jobInstance.id,
                    frame,
                    init_type: 'text',
                    text: textPrompt.trim(),
                }),
            });

            const data = await response.json();
            if (response.ok && data.session_id) {
                setTracking({
                    sessionId: data.session_id,
                    isTracking: true,
                    frameIdx: frame,
                    textPrompt: textPrompt.trim(),
                });

                // Apply initial mask as annotation
                if (data.polygon && data.polygon.length >= 6) {
                    await applyTrackingResult(data.polygon, frame);
                }

                setSuccess(`Started tracking "${textPrompt}"`);
            } else {
                setError(data.error || 'Failed to initialize tracking');
            }
        } catch (err) {
            setError('Failed to start tracking');
        } finally {
            setLoading(false);
        }
    }, [textPrompt, jobInstance.id, frame]);

    // ========== Text-to-Track: Propagate ==========
    const handleTrackFrame = useCallback(async () => {
        if (!tracking.sessionId) {
            setError('No active tracking session');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await authenticatedFetch(`${API_BASE}/track/frame`, {
                method: 'POST',
                body: JSON.stringify({
                    job_id: jobInstance.id,
                    frame,
                    session_id: tracking.sessionId,
                }),
            });

            const data = await response.json();
            if (response.ok && data.polygon) {
                // Apply tracked result
                await applyTrackingResult(data.polygon, frame);

                setTracking((prev) => ({
                    ...prev,
                    frameIdx: frame,
                }));

                setSuccess(`Tracked to frame #${frame}`);
            } else {
                setError(data.error || 'Tracking lost');
            }
        } catch (err) {
            setError('Failed to track frame');
        } finally {
            setLoading(false);
        }
    }, [tracking.sessionId, jobInstance.id, frame]);

    // ========== Stop Tracking ==========
    const handleStopTracking = useCallback(async () => {
        if (tracking.sessionId) {
            try {
                await authenticatedFetch(`${API_BASE}/track/clear`, {
                    method: 'POST',
                    body: JSON.stringify({
                        session_id: tracking.sessionId,
                    }),
                });
            } catch (err) {
                // Ignore errors on cleanup
            }
        }

        setTracking({
            sessionId: null,
            isTracking: false,
            frameIdx: -1,
            textPrompt: null,
        });
        setSuccess('Tracking stopped');
    }, [tracking.sessionId]);

    // ========== Apply Results ==========
    const applyTrackingResult = useCallback(async (polygon: number[], targetFrame: number) => {
        const label = labels.find(
            (l) => l.name.toLowerCase() === tracking.textPrompt?.toLowerCase(),
        ) || labels[0];

        if (!label) {
            setError('No matching label found');
            return;
        }

        const objectState = new core.classes.ObjectState({
            frame: targetFrame,
            label,
            objectType: ObjectType.SHAPE,
            shapeType: ShapeType.POLYGON,
            points: polygon,
            occluded: false,
            source: core.enums.Source.AUTO,
            zOrder: curZOrder,
        });

        await createAnnotations([objectState]);
    }, [labels, curZOrder, createAnnotations, tracking.textPrompt]);

    const handleApplyDetections = useCallback(async () => {
        const selectedDets = detections.filter((d) => d.selected);
        if (selectedDets.length === 0) {
            setError('No detections selected');
            return;
        }

        const objectStates: ObjectState[] = [];

        for (const det of selectedDets) {
            // Find or create label
            let label = labels.find(
                (l) => l.name.toLowerCase() === det.label.toLowerCase(),
            );

            if (!label) {
                // Use first label if no match
                label = labels[0];
                if (!label) {
                    continue;
                }
            }

            const points = outputType === 'polygon' && det.polygon?.length >= 6
                ? det.polygon
                : det.bbox;

            const shapeType = outputType === 'polygon' && det.polygon?.length >= 6
                ? ShapeType.POLYGON
                : ShapeType.RECTANGLE;

            const objectState = new core.classes.ObjectState({
                frame,
                label,
                objectType: ObjectType.SHAPE,
                shapeType,
                points,
                occluded: false,
                source: core.enums.Source.AUTO,
                zOrder: curZOrder,
            });

            objectStates.push(objectState);
        }

        if (objectStates.length > 0) {
            await createAnnotations(objectStates);
            setSuccess(`Applied ${objectStates.length} annotations`);
            setDetections([]);
        }
    }, [detections, labels, frame, curZOrder, createAnnotations, outputType]);

    // Toggle detection selection
    const toggleDetectionSelection = useCallback((idx: number) => {
        setDetections((prev) => prev.map((d, i) => (
            i === idx ? { ...d, selected: !d.selected } : d
        )));
    }, []);

    // Select/deselect all
    const selectAllDetections = useCallback((selected: boolean) => {
        setDetections((prev) => prev.map((d) => ({ ...d, selected })));
    }, []);

    // Clear results
    const handleClear = useCallback(() => {
        setDetections([]);
        setSegmentResult(null);
        setError(null);
        setSuccess(null);
    }, []);

    // Count selected detections
    const selectedCount = useMemo(
        () => detections.filter((d) => d.selected).length,
        [detections],
    );

    return (
        <div className="cvat-sam3-inline-panel" style={{ padding: '8px 0' }}>
            <Spin spinning={loading}>
                {/* Messages */}
                {error && (
                    <Alert
                        message={error}
                        type="error"
                        closable
                        onClose={() => setError(null)}
                        style={{ marginBottom: 8 }}
                    />
                )}
                {success && (
                    <Alert
                        message={success}
                        type="success"
                        closable
                        onClose={() => setSuccess(null)}
                        style={{ marginBottom: 8 }}
                    />
                )}

                {/* Mode Selector */}
                <div style={{ marginBottom: 8 }}>
                    <Radio.Group
                        value={mode}
                        onChange={(e) => {
                            setMode(e.target.value);
                            handleClear();
                        }}
                        size="small"
                        buttonStyle="solid"
                        style={{ width: '100%' }}
                    >
                        <Radio.Button value="segment" style={{ width: '33%', textAlign: 'center' }}>
                            <AimOutlined /> Segment
                        </Radio.Button>
                        <Radio.Button value="detect" style={{ width: '34%', textAlign: 'center' }}>
                            <ScanOutlined /> Detect
                        </Radio.Button>
                        <Radio.Button value="track" style={{ width: '33%', textAlign: 'center' }}>
                            <VideoCameraOutlined /> Track
                        </Radio.Button>
                    </Radio.Group>
                </div>

                {/* Text Prompt Input */}
                <div style={{ marginBottom: 8 }}>
                    <Text style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
                        Text Prompt (e.g., "person", "car", "dog")
                    </Text>
                    <Input
                        placeholder="Enter what to find..."
                        value={textPrompt}
                        onChange={(e) => setTextPrompt(e.target.value)}
                        onPressEnter={() => {
                            if (mode === 'segment') handleSegmentText();
                            else if (mode === 'detect') handleDetect();
                            else if (mode === 'track' && !tracking.isTracking) handleTrackInit();
                        }}
                        prefix={<SearchOutlined />}
                        allowClear
                        disabled={tracking.isTracking}
                    />
                </div>

                {/* Mode-specific controls */}
                {mode === 'segment' && (
                    <>
                        <Button
                            type="primary"
                            icon={<AimOutlined />}
                            onClick={handleSegmentText}
                            disabled={!textPrompt.trim()}
                            block
                            style={{ marginBottom: 8 }}
                        >
                            Segment "{textPrompt || '...'}"
                        </Button>

                        {segmentResult && (
                            <Alert
                                message="Segmentation complete! Click on canvas to refine with points."
                                type="info"
                                style={{ marginBottom: 8 }}
                            />
                        )}
                    </>
                )}

                {mode === 'detect' && (
                    <>
                        {/* Threshold slider */}
                        <div style={{ marginBottom: 8 }}>
                            <Text style={{ fontSize: 11 }}>Confidence: {threshold.toFixed(2)}</Text>
                            <Slider
                                min={0.05}
                                max={0.95}
                                step={0.05}
                                value={threshold}
                                onChange={setThreshold}
                                tooltip={{ formatter: (v) => v?.toFixed(2) }}
                            />
                        </div>

                        {/* Output type */}
                        <div style={{ marginBottom: 8 }}>
                            <Text style={{ fontSize: 11, marginRight: 8 }}>Output:</Text>
                            <Radio.Group
                                value={outputType}
                                onChange={(e) => setOutputType(e.target.value)}
                                size="small"
                            >
                                <Radio value="polygon">Polygon</Radio>
                                <Radio value="rectangle">Rectangle</Radio>
                            </Radio.Group>
                        </div>

                        <Button
                            type="primary"
                            icon={<ScanOutlined />}
                            onClick={handleDetect}
                            disabled={!textPrompt.trim()}
                            block
                            style={{ marginBottom: 8 }}
                        >
                            Detect All "{textPrompt || '...'}"
                        </Button>

                        {/* Detection results */}
                        {detections.length > 0 && (
                            <Card size="small" style={{ marginBottom: 8 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                    <Text strong style={{ fontSize: 12 }}>
                                        Results ({selectedCount}/{detections.length})
                                    </Text>
                                    <Space size={4}>
                                        <Button size="small" onClick={() => selectAllDetections(true)}>
                                            All
                                        </Button>
                                        <Button size="small" onClick={() => selectAllDetections(false)}>
                                            None
                                        </Button>
                                    </Space>
                                </div>

                                <List
                                    size="small"
                                    dataSource={detections}
                                    style={{ maxHeight: 120, overflow: 'auto' }}
                                    renderItem={(det, idx) => (
                                        <List.Item
                                            style={{
                                                padding: '4px',
                                                cursor: 'pointer',
                                                backgroundColor: det.selected ? '#e6f7ff' : 'transparent',
                                            }}
                                            onClick={() => toggleDetectionSelection(idx)}
                                        >
                                            <Space>
                                                <Badge
                                                    status={det.selected ? 'success' : 'default'}
                                                />
                                                <Text style={{ fontSize: 11 }}>
                                                    {det.label} ({(det.score * 100).toFixed(0)}%)
                                                </Text>
                                            </Space>
                                        </List.Item>
                                    )}
                                />

                                <Button
                                    type="primary"
                                    icon={<CheckCircleOutlined />}
                                    onClick={handleApplyDetections}
                                    disabled={selectedCount === 0}
                                    block
                                    style={{ marginTop: 8 }}
                                >
                                    Apply {selectedCount} Annotations
                                </Button>
                            </Card>
                        )}
                    </>
                )}

                {mode === 'track' && (
                    <>
                        {/* Tracking status */}
                        {tracking.isTracking ? (
                            <Card size="small" style={{ marginBottom: 8, backgroundColor: '#f6ffed' }}>
                                <Space direction="vertical" style={{ width: '100%' }}>
                                    <div>
                                        <PlayCircleOutlined style={{ color: '#52c41a', marginRight: 4 }} />
                                        <Text strong>Tracking: "{tracking.textPrompt}"</Text>
                                    </div>
                                    <Text type="secondary" style={{ fontSize: 11 }}>
                                        Started at frame #{tracking.frameIdx}
                                    </Text>

                                    <Space style={{ width: '100%' }}>
                                        <Button
                                            type="primary"
                                            icon={<PlayCircleOutlined />}
                                            onClick={handleTrackFrame}
                                            disabled={frame === tracking.frameIdx}
                                            style={{ flex: 1 }}
                                        >
                                            Track Frame #{frame}
                                        </Button>
                                        <Button
                                            danger
                                            icon={<StopOutlined />}
                                            onClick={handleStopTracking}
                                        >
                                            Stop
                                        </Button>
                                    </Space>
                                </Space>
                            </Card>
                        ) : (
                            <>
                                <Alert
                                    message="Video Tracking"
                                    description="Enter a text prompt to track an object across frames. Start on the first frame where the object appears."
                                    type="info"
                                    style={{ marginBottom: 8 }}
                                />

                                <Button
                                    type="primary"
                                    icon={<VideoCameraOutlined />}
                                    onClick={handleTrackInit}
                                    disabled={!textPrompt.trim()}
                                    block
                                >
                                    Start Tracking "{textPrompt || '...'}"
                                </Button>
                            </>
                        )}
                    </>
                )}

                {/* Clear button */}
                {(detections.length > 0 || segmentResult) && (
                    <Button
                        icon={<ClearOutlined />}
                        onClick={handleClear}
                        block
                        style={{ marginTop: 8 }}
                    >
                        Clear Results
                    </Button>
                )}
            </Spin>
        </div>
    );
}

export default React.memo(SAM3InlinePanel);
