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
    Modal,
    Button,
    Checkbox,
    Slider,
    Radio,
    Space,
    Typography,
    Divider,
    Alert,
    Spin,
    InputNumber,
    Tooltip,
    Badge,
    Card,
    Row,
    Col,
    message,
} from 'antd';
import {
    ThunderboltOutlined,
    CheckCircleOutlined,
    InfoCircleOutlined,
    ClearOutlined,
    ReloadOutlined,
    EyeOutlined,
} from '@ant-design/icons';

const { Text, Title } = Typography;

// API helper functions
const API_BASE = '/api/lambda/yoloe';

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

interface PredictionResult {
    frame: number;
    detections: {
        label: string;
        points: number[];
        type: string;
        confidence: string;
    }[];
}

// Maximum number of reference frames
const MAX_REFERENCES = 50;

// Output type options
const OUTPUT_TYPES = [
    { label: 'Bounding Box', value: 'rectangle' },
    { label: 'Polygon (Segmentation)', value: 'polygon' },
    { label: 'OBB (Rotated Box)', value: 'obb' },
];

interface YOLOEVisualPromptPanelProps {
    jobId: number;
    currentFrame: number;
    totalFrames: number;
    onApplyAnnotations: (results: PredictionResult[]) => void;
    onClose: () => void;
}

const YOLOEVisualPromptPanel: React.FC<YOLOEVisualPromptPanelProps> = ({
    jobId,
    currentFrame,
    totalFrames,
    onApplyAnnotations,
    onClose,
}) => {
    // State
    const [loading, setLoading] = useState(false);
    const [annotatedFrames, setAnnotatedFrames] = useState<AnnotatedFrame[]>([]);
    const [selectedFrames, setSelectedFrames] = useState<number[]>([]);
    const [vpeStatus, setVPEStatus] = useState<VPEStatus | null>(null);
    const [threshold, setThreshold] = useState(0.25);
    const [outputType, setOutputType] = useState<string>('rectangle');
    const [targetMode, setTargetMode] = useState<'current' | 'range'>('current');
    const [rangeStart, setRangeStart] = useState(0);
    const [rangeEnd, setRangeEnd] = useState(totalFrames - 1);
    const [predictions, setPredictions] = useState<PredictionResult[] | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Fetch annotated frames on mount
    useEffect(() => {
        fetchAnnotatedFrames();
        fetchVPEStatus();
    }, [jobId]);

    const fetchAnnotatedFrames = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${API_BASE}/annotated-frames?job_id=${jobId}`);
            const data = await response.json();
            if (response.ok) {
                setAnnotatedFrames(data.frames || []);
            } else {
                setError(data.error || 'Failed to fetch annotated frames');
            }
        } catch (err) {
            setError('Failed to connect to server');
        } finally {
            setLoading(false);
        }
    };

    const fetchVPEStatus = async () => {
        try {
            const response = await fetch(`${API_BASE}/status?job_id=${jobId}`);
            const data = await response.json();
            if (response.ok) {
                setVPEStatus(data);
                if (data.reference_frames) {
                    setSelectedFrames(data.reference_frames);
                }
            }
        } catch (err) {
            console.warn('Failed to fetch VPE status');
        }
    };

    const handleGenerateVPE = async () => {
        if (selectedFrames.length === 0) {
            message.warning('Please select at least one reference frame');
            return;
        }

        try {
            setLoading(true);
            setError(null);

            const response = await fetch(`${API_BASE}/generate-vpe`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    reference_frames: selectedFrames,
                }),
            });

            const data = await response.json();
            if (response.ok) {
                message.success(`VPE generated from ${data.num_references} references with ${data.total_annotations} annotations`);
                await fetchVPEStatus();
            } else {
                setError(data.error || 'Failed to generate VPE');
            }
        } catch (err) {
            setError('Failed to generate VPE');
        } finally {
            setLoading(false);
        }
    };

    const handlePredict = async () => {
        if (!vpeStatus?.exists) {
            message.warning('Please generate VPE first');
            return;
        }

        try {
            setLoading(true);
            setError(null);

            // Determine target frames
            let targetFrames: number[];
            if (targetMode === 'current') {
                targetFrames = [currentFrame];
            } else {
                // Generate range, excluding reference frames
                targetFrames = [];
                for (let i = rangeStart; i <= rangeEnd; i++) {
                    if (!selectedFrames.includes(i)) {
                        targetFrames.push(i);
                    }
                }
            }

            if (targetFrames.length === 0) {
                message.warning('No target frames to process');
                return;
            }

            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    frames: targetFrames,
                    threshold,
                    output_type: outputType,
                }),
            });

            const data = await response.json();
            if (response.ok) {
                setPredictions(data.results || []);
                const totalDetections = data.results?.reduce(
                    (acc: number, r: PredictionResult) => acc + r.detections.length,
                    0
                ) || 0;
                message.success(`Found ${totalDetections} detections in ${data.results?.length || 0} frames`);
            } else {
                setError(data.error || 'Failed to run prediction');
            }
        } catch (err) {
            setError('Failed to run prediction');
        } finally {
            setLoading(false);
        }
    };

    const handleApply = async () => {
        if (!predictions || predictions.length === 0) {
            message.warning('No predictions to apply');
            return;
        }

        try {
            setLoading(true);
            setError(null);

            const response = await fetch(`${API_BASE}/apply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    results: predictions,
                }),
            });

            const data = await response.json();
            if (response.ok) {
                message.success(`Applied ${data.applied} annotations`);
                onApplyAnnotations(predictions);
                setPredictions(null);
            } else {
                setError(data.error || 'Failed to apply annotations');
            }
        } catch (err) {
            setError('Failed to apply annotations');
        } finally {
            setLoading(false);
        }
    };

    const handleClearVPE = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${API_BASE}/clear`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId }),
            });

            if (response.ok) {
                message.success('VPE cache cleared');
                setVPEStatus(null);
                setSelectedFrames([]);
                setPredictions(null);
            }
        } catch (err) {
            setError('Failed to clear VPE');
        } finally {
            setLoading(false);
        }
    };

    const handleFrameSelect = (frame: number, checked: boolean) => {
        if (checked) {
            if (selectedFrames.length >= MAX_REFERENCES) {
                message.warning(`Maximum ${MAX_REFERENCES} reference frames allowed`);
                return;
            }
            setSelectedFrames([...selectedFrames, frame]);
        } else {
            setSelectedFrames(selectedFrames.filter(f => f !== frame));
        }
    };

    const handleSelectAll = () => {
        const framesToSelect = annotatedFrames
            .slice(0, MAX_REFERENCES)
            .map(f => f.frame);
        setSelectedFrames(framesToSelect);
    };

    const handleDeselectAll = () => {
        setSelectedFrames([]);
    };

    // Get unique labels from selected frames
    const selectedLabels = useMemo(() => {
        const labels = new Set<string>();
        annotatedFrames
            .filter(f => selectedFrames.includes(f.frame))
            .forEach(f => f.labels.forEach(l => labels.add(l)));
        return Array.from(labels).sort();
    }, [annotatedFrames, selectedFrames]);

    return (
        <div style={{ padding: '16px', maxHeight: '80vh', overflow: 'auto' }}>
            <Spin spinning={loading}>
                {error && (
                    <Alert
                        message="Error"
                        description={error}
                        type="error"
                        closable
                        onClose={() => setError(null)}
                        style={{ marginBottom: 16 }}
                    />
                )}

                {/* VPE Status */}
                {vpeStatus?.exists && (
                    <Card size="small" style={{ marginBottom: 16 }}>
                        <Space>
                            <CheckCircleOutlined style={{ color: '#52c41a' }} />
                            <Text>
                                VPE cached: {vpeStatus.num_references} references,{' '}
                                {vpeStatus.total_annotations} annotations,{' '}
                                expires in {vpeStatus.ttl_remaining_days?.toFixed(1)} days
                            </Text>
                            <Button
                                size="small"
                                icon={<ClearOutlined />}
                                onClick={handleClearVPE}
                            >
                                Clear
                            </Button>
                        </Space>
                    </Card>
                )}

                {/* Reference Frame Selection */}
                <Title level={5}>
                    <EyeOutlined /> Reference Frames
                    <Badge
                        count={`${selectedFrames.length}/${MAX_REFERENCES}`}
                        style={{ backgroundColor: selectedFrames.length >= MAX_REFERENCES ? '#ff4d4f' : '#52c41a', marginLeft: 8 }}
                    />
                </Title>
                <Text type="secondary">
                    Select annotated frames to use as visual references. The model will learn from these examples.
                </Text>

                <div style={{ margin: '8px 0' }}>
                    <Space>
                        <Button size="small" onClick={handleSelectAll}>Select All</Button>
                        <Button size="small" onClick={handleDeselectAll}>Deselect All</Button>
                        <Button
                            size="small"
                            icon={<ReloadOutlined />}
                            onClick={fetchAnnotatedFrames}
                        >
                            Refresh
                        </Button>
                    </Space>
                </div>

                <div style={{ maxHeight: 200, overflow: 'auto', border: '1px solid #d9d9d9', padding: 8, marginBottom: 16 }}>
                    {annotatedFrames.length === 0 ? (
                        <Text type="secondary">No annotated frames found. Annotate some frames first.</Text>
                    ) : (
                        annotatedFrames.map(frame => (
                            <div key={frame.frame} style={{ padding: '4px 0' }}>
                                <Checkbox
                                    checked={selectedFrames.includes(frame.frame)}
                                    onChange={e => handleFrameSelect(frame.frame, e.target.checked)}
                                    disabled={!selectedFrames.includes(frame.frame) && selectedFrames.length >= MAX_REFERENCES}
                                >
                                    Frame {frame.frame}
                                    <Text type="secondary" style={{ marginLeft: 8 }}>
                                        ({frame.annotation_count} annotations: {frame.labels.join(', ')})
                                    </Text>
                                </Checkbox>
                            </div>
                        ))
                    )}
                </div>

                {selectedLabels.length > 0 && (
                    <Alert
                        message={`Classes: ${selectedLabels.join(', ')}`}
                        type="info"
                        style={{ marginBottom: 16 }}
                    />
                )}

                <Button
                    type="primary"
                    icon={<ThunderboltOutlined />}
                    onClick={handleGenerateVPE}
                    disabled={selectedFrames.length === 0}
                    style={{ marginBottom: 16 }}
                >
                    Generate VPE from {selectedFrames.length} References
                </Button>

                <Divider />

                {/* Prediction Settings */}
                <Title level={5}>Detection Settings</Title>

                <Row gutter={16}>
                    <Col span={12}>
                        <Text>Confidence Threshold</Text>
                        <Slider
                            min={0.05}
                            max={0.95}
                            step={0.05}
                            value={threshold}
                            onChange={setThreshold}
                            marks={{ 0.1: '0.1', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75' }}
                        />
                    </Col>
                    <Col span={12}>
                        <Text>Output Type</Text>
                        <Radio.Group
                            value={outputType}
                            onChange={e => setOutputType(e.target.value)}
                            style={{ display: 'block', marginTop: 8 }}
                        >
                            {OUTPUT_TYPES.map(opt => (
                                <Radio key={opt.value} value={opt.value} style={{ display: 'block' }}>
                                    {opt.label}
                                </Radio>
                            ))}
                        </Radio.Group>
                    </Col>
                </Row>

                <Divider />

                {/* Target Selection */}
                <Title level={5}>Target Frames</Title>
                <Radio.Group
                    value={targetMode}
                    onChange={e => setTargetMode(e.target.value)}
                    style={{ marginBottom: 16 }}
                >
                    <Radio value="current">Current Frame ({currentFrame})</Radio>
                    <Radio value="range">Frame Range</Radio>
                </Radio.Group>

                {targetMode === 'range' && (
                    <Space style={{ marginBottom: 16 }}>
                        <Text>From:</Text>
                        <InputNumber
                            min={0}
                            max={totalFrames - 1}
                            value={rangeStart}
                            onChange={v => setRangeStart(v || 0)}
                        />
                        <Text>To:</Text>
                        <InputNumber
                            min={0}
                            max={totalFrames - 1}
                            value={rangeEnd}
                            onChange={v => setRangeEnd(v || totalFrames - 1)}
                        />
                    </Space>
                )}

                <div style={{ marginTop: 16 }}>
                    <Space>
                        <Button
                            type="primary"
                            icon={<ThunderboltOutlined />}
                            onClick={handlePredict}
                            disabled={!vpeStatus?.exists}
                        >
                            Run Detection
                        </Button>
                        {predictions && predictions.length > 0 && (
                            <Button
                                type="primary"
                                icon={<CheckCircleOutlined />}
                                onClick={handleApply}
                                style={{ backgroundColor: '#52c41a', borderColor: '#52c41a' }}
                            >
                                Apply {predictions.reduce((acc, r) => acc + r.detections.length, 0)} Annotations
                            </Button>
                        )}
                    </Space>
                </div>

                {/* Preview Results */}
                {predictions && predictions.length > 0 && (
                    <>
                        <Divider />
                        <Title level={5}>Detection Results</Title>
                        <div style={{ maxHeight: 200, overflow: 'auto' }}>
                            {predictions.map(result => (
                                <div key={result.frame} style={{ marginBottom: 8 }}>
                                    <Text strong>Frame {result.frame}:</Text>
                                    <Text> {result.detections.length} detections</Text>
                                    <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                                        {result.detections.slice(0, 5).map((det, idx) => (
                                            <li key={idx}>
                                                <Text type="secondary">
                                                    {det.label} ({(parseFloat(det.confidence) * 100).toFixed(1)}%)
                                                </Text>
                                            </li>
                                        ))}
                                        {result.detections.length > 5 && (
                                            <li><Text type="secondary">... and {result.detections.length - 5} more</Text></li>
                                        )}
                                    </ul>
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </Spin>
        </div>
    );
};

// Types for plugin integration
interface CVATCore {
    server: {
        request: (url: string, options?: any) => Promise<any>;
    };
    plugins: {
        register: (plugin: any) => void;
    };
}

interface ComponentBuilderArgs {
    dispatch: any;
    actionCreators: any;
    core: CVATCore;
    store: any;
}

type ComponentBuilder = (args: ComponentBuilderArgs) => {
    name: string;
    destructor: () => void;
    globalStateDidUpdate?: (state: any) => void;
};

type PluginEntryPoint = (builder: ComponentBuilder) => void;

// Plugin data storage
interface YOLOEPluginData {
    core: CVATCore | null;
    modalVisible: boolean;
    currentJobId: number | null;
    currentFrame: number;
    totalFrames: number;
}

const pluginData: YOLOEPluginData = {
    core: null,
    modalVisible: false,
    currentJobId: null,
    currentFrame: 0,
    totalFrames: 0,
};

// Create a wrapper component that can be rendered
const YOLOEModalWrapper: React.FC = () => {
    const [visible, setVisible] = React.useState(false);
    const [jobId, setJobId] = React.useState<number | null>(null);
    const [currentFrame, setCurrentFrame] = React.useState(0);
    const [totalFrames, setTotalFrames] = React.useState(0);

    // Listen for custom events to show/hide modal
    React.useEffect(() => {
        const handleOpenModal = (e: CustomEvent) => {
            const { jobId: jId, currentFrame: cf, totalFrames: tf } = e.detail;
            setJobId(jId);
            setCurrentFrame(cf);
            setTotalFrames(tf);
            setVisible(true);
        };

        const handleCloseModal = () => {
            setVisible(false);
        };

        window.addEventListener('yoloe.openModal' as any, handleOpenModal);
        window.addEventListener('yoloe.closeModal' as any, handleCloseModal);

        return () => {
            window.removeEventListener('yoloe.openModal' as any, handleOpenModal);
            window.removeEventListener('yoloe.closeModal' as any, handleCloseModal);
        };
    }, []);

    if (!visible || jobId === null) return null;

    return (
        <Modal
            title="YOLOE Visual Prompting"
            open={visible}
            onCancel={() => setVisible(false)}
            footer={null}
            width={700}
            style={{ top: 50 }}
        >
            <YOLOEVisualPromptPanel
                jobId={jobId}
                currentFrame={currentFrame}
                totalFrames={totalFrames}
                onApplyAnnotations={(results: PredictionResult[]) => {
                    console.log('Applied annotations:', results);
                    // Trigger annotation refresh
                    window.dispatchEvent(new CustomEvent('yoloe.annotationsApplied', { detail: { results } }));
                }}
                onClose={() => setVisible(false)}
            />
        </Modal>
    );
};

// Plugin builder
const builder: ComponentBuilder = ({ core }) => {
    pluginData.core = core;

    // Register the plugin API wrapper
    const yoloePlugin = {
        name: 'YOLOE Visual Prompt',
        description: 'Enables YOLOE Visual Prompting for detection based on annotated references',
        cvat: {
            // Hook into lambda calls if needed
            lambda: {
                call: {
                    async enter(
                        plugin: any,
                        taskID: number,
                        model: any,
                        args: any,
                    ): Promise<null | { preventMethodCall: boolean }> {
                        // Check if this is a YOLOE model call
                        if (model?.id?.includes('yoloe-visual-prompt')) {
                            // The YOLOE visual prompt flow is different
                            // It goes through our custom API, not lambda.call
                            console.log('YOLOE Visual Prompt detected in lambda.call');
                        }
                        return null;
                    },
                },
            },
        },
        data: pluginData,
    };

    core.plugins.register(yoloePlugin);

    return {
        name: 'YOLOE Visual Prompt',
        destructor: () => {
            console.log('YOLOE Visual Prompt plugin destroyed');
        },
        globalStateDidUpdate: (state: any) => {
            // Track current job and frame for the modal
            if (state?.annotation?.job?.instance) {
                pluginData.currentJobId = state.annotation.job.instance.id;
                pluginData.totalFrames = state.annotation.job.instance.stopFrame -
                                         state.annotation.job.instance.startFrame + 1;
            }
            if (state?.annotation?.player?.frame?.number !== undefined) {
                pluginData.currentFrame = state.annotation.player.frame.number;
            }
        },
    };
};

// Helper function to open the YOLOE modal programmatically
export function openYOLOEModal(jobId: number, currentFrame: number, totalFrames: number): void {
    window.dispatchEvent(new CustomEvent('yoloe.openModal', {
        detail: { jobId, currentFrame, totalFrames },
    }));
}

// Helper function to close the modal
export function closeYOLOEModal(): void {
    window.dispatchEvent(new CustomEvent('yoloe.closeModal'));
}

// Register the plugin when the document is ready
function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any).cvatUI.registerComponent(builder);
    }
}

window.addEventListener('plugins.ready', register, { once: true });

export default register;
export { YOLOEVisualPromptPanel, YOLOEModalWrapper };
export type { YOLOEVisualPromptPanelProps, PredictionResult };
