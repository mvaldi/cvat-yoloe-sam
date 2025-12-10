// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, {
    useState,
    useEffect,
    useCallback,
    useMemo,
} from 'react';
import {
    Button,
    Checkbox,
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
    Radio,
} from 'antd';
import {
    ThunderboltOutlined,
    CheckCircleOutlined,
    SyncOutlined,
    ClearOutlined,
    EyeOutlined,
    ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useSelector } from 'react-redux';

import { CombinedState } from 'reducers';
import { Job, Label, ObjectState, ObjectType, ShapeType } from 'cvat-core-wrapper';
import { getCore } from 'cvat-core-wrapper';

const { Text, Title } = Typography;
const core = getCore();

// API endpoints
const API_BASE = '/api/lambda/yoloe';

// Max references
const MAX_REFERENCES = 50;

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

    // Add CSRF token for non-GET requests
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

interface AnnotatedFrame {
    frame: number;
    annotation_count: number;
    labels: string[];
}

interface VPEStatus {
    exists: boolean;
    job_id: number;
    reference_frames?: number[];
    class_names?: string[];
    num_references?: number;
    total_annotations?: number;
    ttl_remaining_days?: number;
}

interface Detection {
    label: string;
    points: number[];
    type: string;
    confidence: string;
}

interface PredictionResult {
    frame: number;
    detections: Detection[];
}

interface Props {
    jobInstance: Job;
    frame: number;
    labels: Label[];
    curZOrder: number;
    createAnnotations: (states: ObjectState[]) => Promise<void>;
}

function YOLOEInlinePanel(props: Props): JSX.Element {
    const {
        jobInstance,
        frame,
        labels,
        curZOrder,
        createAnnotations,
    } = props;

    // Get annotation states to detect changes
    const annotationStates = useSelector((state: CombinedState) => state.annotation.annotations.states);

    // State
    const [loading, setLoading] = useState(false);
    const [annotatedFrames, setAnnotatedFrames] = useState<AnnotatedFrame[]>([]);
    const [selectedFrames, setSelectedFrames] = useState<number[]>([]);
    const [vpeStatus, setVPEStatus] = useState<VPEStatus | null>(null);
    const [threshold, setThreshold] = useState(0.25);
    const [error, setError] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<PredictionResult | null>(null);
    const [showPreview, setShowPreview] = useState(false);
    const [outputType, setOutputType] = useState<'rectangle' | 'polygon' | 'obb'>('rectangle');

    // Session storage key for selected frames
    const sessionKey = useMemo(() => `yoloe_refs_${jobInstance.id}`, [jobInstance.id]);

    // Load selected frames from session storage
    useEffect(() => {
        const stored = sessionStorage.getItem(sessionKey);
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed)) {
                    setSelectedFrames(parsed);
                }
            } catch (e) {
                // Ignore parse errors
            }
        }
    }, [sessionKey]);

    // Save selected frames to session storage
    useEffect(() => {
        sessionStorage.setItem(sessionKey, JSON.stringify(selectedFrames));
    }, [selectedFrames, sessionKey]);

    // Fetch annotated frames
    const fetchAnnotatedFrames = useCallback(async () => {
        try {
            const response = await authenticatedFetch(`${API_BASE}/annotated-frames?job_id=${jobInstance.id}`);
            const data = await response.json();
            if (response.ok) {
                setAnnotatedFrames(data.frames || []);
                // Remove selected frames that are no longer annotated
                const validFrames = new Set<number>((data.frames || []).map((f: AnnotatedFrame) => f.frame));
                setSelectedFrames((prev: number[]) => prev.filter((f: number) => validFrames.has(f)));
            } else {
                setError(data.error || 'Failed to fetch annotated frames');
            }
        } catch (err) {
            setError('Failed to connect to server');
        }
    }, [jobInstance.id]);

    // Fetch VPE status
    const fetchVPEStatus = useCallback(async () => {
        try {
            const response = await authenticatedFetch(`${API_BASE}/status?job_id=${jobInstance.id}`);
            const data = await response.json();
            if (response.ok) {
                setVPEStatus(data);
            }
        } catch (err) {
            // Silently fail for status check
        }
    }, [jobInstance.id]);

    // Initial fetch
    useEffect(() => {
        fetchAnnotatedFrames();
        fetchVPEStatus();
    }, [fetchAnnotatedFrames, fetchVPEStatus]);

    // Auto-refresh when annotations change (detect new annotations)
    useEffect(() => {
        // Debounce the refresh to avoid too many calls
        const timeoutId = setTimeout(() => {
            fetchAnnotatedFrames();
        }, 500);

        return () => clearTimeout(timeoutId);
    }, [annotationStates, fetchAnnotatedFrames]);

    // Handle frame selection
    const handleFrameSelect = useCallback((frameNum: number, checked: boolean) => {
        if (checked) {
            if (selectedFrames.length >= MAX_REFERENCES) {
                Modal.warning({
                    title: 'Maximum references reached',
                    content: `You can select up to ${MAX_REFERENCES} reference frames.`,
                });
                return;
            }
            setSelectedFrames((prev: number[]) => [...prev, frameNum]);
        } else {
            setSelectedFrames((prev: number[]) => prev.filter((f: number) => f !== frameNum));
        }
    }, [selectedFrames.length]);

    // Select all
    const handleSelectAll = useCallback(() => {
        const framesToSelect = annotatedFrames
            .slice(0, MAX_REFERENCES)
            .map((f: AnnotatedFrame) => f.frame);
        setSelectedFrames(framesToSelect);
    }, [annotatedFrames]);

    // Clear selection
    const handleClearSelection = useCallback(() => {
        setSelectedFrames([]);
    }, []);

    // Generate VPE
    const handleGenerateVPE = useCallback(async () => {
        if (selectedFrames.length === 0) {
            Modal.warning({
                title: 'No frames selected',
                content: 'Please select at least one annotated frame as reference.',
            });
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await authenticatedFetch(`${API_BASE}/generate-vpe`, {
                method: 'POST',
                body: JSON.stringify({
                    job_id: jobInstance.id,
                    reference_frames: selectedFrames,
                }),
            });

            const data = await response.json();
            if (response.ok) {
                Modal.success({
                    title: 'VPE Generated',
                    content: `Generated from ${data.num_references} frames with ${data.total_annotations} annotations.`,
                });
                await fetchVPEStatus();
            } else {
                setError(data.error || 'Failed to generate VPE');
            }
        } catch (err) {
            setError('Failed to generate VPE');
        } finally {
            setLoading(false);
        }
    }, [selectedFrames, jobInstance.id, fetchVPEStatus]);

    // Detect current frame
    const handleDetect = useCallback(async () => {
        if (!vpeStatus?.exists) {
            Modal.warning({
                title: 'VPE not ready',
                content: 'Please generate VPE first by selecting reference frames.',
            });
            return;
        }

        // Check if current frame is a reference frame
        if (selectedFrames.includes(frame)) {
            Modal.warning({
                title: 'Reference frame',
                content: 'Current frame is a reference frame. Navigate to a different frame to detect.',
            });
            return;
        }

        setLoading(true);
        setError(null);
        setPredictions(null);

        try {
            const response = await authenticatedFetch(`${API_BASE}/predict`, {
                method: 'POST',
                body: JSON.stringify({
                    job_id: jobInstance.id,
                    frames: [frame],
                    threshold,
                    output_type: outputType,
                }),
            });

            const data = await response.json();
            if (response.ok && data.results?.length > 0) {
                const result = data.results[0];
                if (result.detections.length > 0) {
                    setPredictions(result);
                    setShowPreview(true);
                } else {
                    Modal.info({
                        title: 'No detections',
                        content: `No objects detected with confidence >= ${threshold}. Try lowering the threshold.`,
                    });
                }
            } else {
                setError(data.error || 'Failed to run detection');
            }
        } catch (err) {
            setError('Failed to run detection');
        } finally {
            setLoading(false);
        }
    }, [vpeStatus, selectedFrames, frame, jobInstance.id, threshold, outputType]);

    // Apply predictions
    const handleApplyPredictions = useCallback(async () => {
        if (!predictions) return;

        const objectStates: ObjectState[] = [];

        for (const det of predictions.detections) {
            // Find matching label in job
            const label = labels.find(
                (l) => l.name.toLowerCase() === det.label.toLowerCase(),
            );

            if (!label) {
                console.warn(`Label "${det.label}" not found in job labels`);
                continue;
            }

            // Create ObjectState
            const objectState = new core.classes.ObjectState({
                frame: predictions.frame,
                label,
                objectType: ObjectType.SHAPE,
                shapeType: det.type as ShapeType,
                points: det.points,
                occluded: false,
                source: core.enums.Source.AUTO,
                zOrder: curZOrder,
            });

            objectStates.push(objectState);
        }

        if (objectStates.length > 0) {
            await createAnnotations(objectStates);
        }

        setShowPreview(false);
        setPredictions(null);
    }, [predictions, labels, curZOrder, createAnnotations]);

    // Cancel preview
    const handleCancelPreview = useCallback(() => {
        setShowPreview(false);
        setPredictions(null);
    }, []);

    // Check if current frame is already annotated (reference)
    const isCurrentFrameReference = useMemo(
        () => selectedFrames.includes(frame),
        [selectedFrames, frame],
    );

    // Get labels from selected frames
    const selectedLabels = useMemo(() => {
        const labelSet = new Set<string>();
        annotatedFrames
            .filter((f: AnnotatedFrame) => selectedFrames.includes(f.frame))
            .forEach((f: AnnotatedFrame) => f.labels.forEach((l: string) => labelSet.add(l)));
        return Array.from(labelSet).sort();
    }, [annotatedFrames, selectedFrames]);

    return (
        <div className="cvat-yoloe-inline-panel" style={{ padding: '8px 0' }}>
            <Spin spinning={loading}>
                {error && (
                    <Alert
                        message={error}
                        type="error"
                        closable
                        onClose={() => setError(null)}
                        style={{ marginBottom: 8 }}
                    />
                )}

                {/* VPE Status */}
                {vpeStatus?.exists && (
                    <div style={{ marginBottom: 8, padding: '4px 8px', background: '#f6ffed', borderRadius: 4 }}>
                        <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 4 }} />
                        <Text type="success" style={{ fontSize: 12 }}>
                            VPE ready ({vpeStatus.num_references} refs, {vpeStatus.class_names?.length} classes)
                        </Text>
                    </div>
                )}

                {/* Reference Frames Section */}
                <div style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <Text strong style={{ fontSize: 12 }}>
                            Reference Frames
                            <Badge
                                count={`${selectedFrames.length}/${MAX_REFERENCES}`}
                                style={{
                                    backgroundColor: selectedFrames.length >= MAX_REFERENCES ? '#ff4d4f' : '#1890ff',
                                    marginLeft: 8,
                                    fontSize: 10,
                                }}
                            />
                        </Text>
                        <Space size={4}>
                            <Tooltip title="Refresh list">
                                <Button
                                    size="small"
                                    type="text"
                                    icon={<SyncOutlined />}
                                    onClick={fetchAnnotatedFrames}
                                />
                            </Tooltip>
                        </Space>
                    </div>

                    {/* Frame list */}
                    <div
                        style={{
                            maxHeight: 150,
                            overflow: 'auto',
                            border: '1px solid #d9d9d9',
                            borderRadius: 4,
                            padding: 4,
                        }}
                    >
                        {annotatedFrames.length === 0 ? (
                            <Empty
                                image={Empty.PRESENTED_IMAGE_SIMPLE}
                                description="No annotated frames"
                                style={{ margin: '8px 0' }}
                            />
                        ) : (
                            <List
                                size="small"
                                dataSource={annotatedFrames}
                                renderItem={(item: AnnotatedFrame) => (
                                    <List.Item style={{ padding: '2px 4px' }}>
                                        <Checkbox
                                            checked={selectedFrames.includes(item.frame)}
                                            onChange={(e: { target: { checked: boolean } }) => handleFrameSelect(item.frame, e.target.checked)}
                                            disabled={
                                                !selectedFrames.includes(item.frame) &&
                                                selectedFrames.length >= MAX_REFERENCES
                                            }
                                        >
                                            <Text style={{ fontSize: 11 }}>
                                                <span><strong>#{item.frame}</strong></span>
                                                <Text type="secondary" style={{ marginLeft: 4 }}>
                                                    ({item.annotation_count}) {item.labels.slice(0, 2).join(', ')}
                                                    {item.labels.length > 2 && '...'}
                                                </Text>
                                            </Text>
                                        </Checkbox>
                                    </List.Item>
                                )}
                            />
                        )}
                    </div>

                    {/* Selection actions */}
                    <div style={{ marginTop: 4 }}>
                        <Space size={4}>
                            <Button size="small" onClick={handleSelectAll} disabled={annotatedFrames.length === 0}>
                                Select All
                            </Button>
                            <Button size="small" onClick={handleClearSelection} disabled={selectedFrames.length === 0}>
                                Clear
                            </Button>
                        </Space>
                    </div>
                </div>

                {/* Selected labels info */}
                {selectedLabels.length > 0 && (
                    <div style={{ marginBottom: 8, fontSize: 11 }}>
                        <Text type="secondary">
                            Classes: {selectedLabels.join(', ')}
                        </Text>
                    </div>
                )}

                {/* Generate VPE button */}
                <Button
                    type="primary"
                    size="small"
                    icon={<ThunderboltOutlined />}
                    onClick={handleGenerateVPE}
                    disabled={selectedFrames.length === 0}
                    block
                    style={{ marginBottom: 8 }}
                >
                    Generate VPE ({selectedFrames.length} frames)
                </Button>

                {/* Threshold slider */}
                <div style={{ marginBottom: 8 }}>
                    <Text style={{ fontSize: 11 }}>Confidence: {threshold.toFixed(2)}</Text>
                    <Slider
                        min={0.05}
                        max={0.95}
                        step={0.05}
                        value={threshold}
                        onChange={setThreshold}
                        tooltip={{ formatter: (v: number | undefined) => v?.toFixed(2) }}
                    />
                </div>

                {/* Output type selector */}
                <div style={{ marginBottom: 8 }}>
                    <Text style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>Output Type:</Text>
                    <Radio.Group
                        value={outputType}
                        onChange={(e) => setOutputType(e.target.value)}
                        size="small"
                        optionType="button"
                        buttonStyle="solid"
                    >
                        <Radio.Button value="rectangle">Rectangle</Radio.Button>
                        <Radio.Button value="polygon">Polygon</Radio.Button>
                        <Radio.Button value="obb">OBB</Radio.Button>
                    </Radio.Group>
                </div>

                {/* Detect button */}
                <Button
                    type="primary"
                    size="small"
                    icon={<EyeOutlined />}
                    onClick={handleDetect}
                    disabled={!vpeStatus?.exists || isCurrentFrameReference}
                    block
                >
                    {isCurrentFrameReference ? 'Current frame is reference' : `Detect Frame #${frame}`}
                </Button>

                {/* Preview Modal */}
                <Modal
                    title={`Detection Preview - Frame #${predictions?.frame}`}
                    open={showPreview}
                    onOk={handleApplyPredictions}
                    onCancel={handleCancelPreview}
                    okText={`Apply ${predictions?.detections.length || 0} annotations`}
                    cancelText="Cancel"
                    width={400}
                >
                    {predictions && (
                        <div>
                            <Alert
                                message={`Found ${predictions.detections.length} objects`}
                                type="info"
                                style={{ marginBottom: 8 }}
                            />
                            <List
                                size="small"
                                dataSource={predictions.detections}
                                renderItem={(det: Detection, idx: number) => (
                                    <List.Item key={idx}>
                                        <Text>
                                            <Text strong>{det.label}</Text>
                                            <Text type="secondary" style={{ marginLeft: 8 }}>
                                                {(parseFloat(det.confidence) * 100).toFixed(1)}%
                                            </Text>
                                        </Text>
                                    </List.Item>
                                )}
                                style={{ maxHeight: 200, overflow: 'auto' }}
                            />
                        </div>
                    )}
                </Modal>
            </Spin>
        </div>
    );
}

export default React.memo(YOLOEInlinePanel);
