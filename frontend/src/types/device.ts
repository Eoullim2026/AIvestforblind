export interface BoundingBox {
    id: string;
    label: string;
    confidence: string;
    x: string;
    y: string;
    w: string;
    h: string;
    color: string;
}

export interface DetectionLog {
    id: string | number;
    name: string;
    direction: string;
    distance: string;
    color: string;
    time?: string;
    confidence?: string;
}

export interface DangerAlert {
    message: string;
    color: string;
}
