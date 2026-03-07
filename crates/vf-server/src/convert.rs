// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Conversion helpers between proto types and vf-core types.

use std::collections::HashMap;

use crate::proto::swarndb::v1 as pb;
use vf_core::types::{
    DistanceMetricType, Metadata as CoreMetadata, MetadataValue as CoreMetadataValue,
};

/// Convert a proto MetadataValue to a core MetadataValue.
pub fn proto_to_core_metadata_value(pv: &pb::MetadataValue) -> Option<CoreMetadataValue> {
    use pb::metadata_value::Value;
    match pv.value.as_ref()? {
        Value::StringValue(s) => Some(CoreMetadataValue::String(s.clone())),
        Value::IntValue(i) => Some(CoreMetadataValue::Int(*i)),
        Value::FloatValue(f) => Some(CoreMetadataValue::Float(*f)),
        Value::BoolValue(b) => Some(CoreMetadataValue::Bool(*b)),
        Value::StringListValue(sl) => Some(CoreMetadataValue::StringList(sl.values.clone())),
    }
}

/// Convert a core MetadataValue to a proto MetadataValue.
pub fn core_to_proto_metadata_value(cv: &CoreMetadataValue) -> pb::MetadataValue {
    use pb::metadata_value::Value;
    let value = match cv {
        CoreMetadataValue::String(s) => Value::StringValue(s.clone()),
        CoreMetadataValue::Int(i) => Value::IntValue(*i),
        CoreMetadataValue::Float(f) => Value::FloatValue(*f),
        CoreMetadataValue::Bool(b) => Value::BoolValue(*b),
        CoreMetadataValue::StringList(sl) => {
            Value::StringListValue(pb::StringList { values: sl.clone() })
        }
    };
    pb::MetadataValue { value: Some(value) }
}

/// Convert proto Metadata to core Metadata.
pub fn proto_to_core_metadata(pm: &pb::Metadata) -> CoreMetadata {
    let mut result = HashMap::new();
    for (key, value) in &pm.fields {
        if let Some(cv) = proto_to_core_metadata_value(value) {
            result.insert(key.clone(), cv);
        }
    }
    result
}

/// Convert core Metadata to proto Metadata.
pub fn core_to_proto_metadata(cm: &CoreMetadata) -> pb::Metadata {
    let mut fields = HashMap::new();
    for (key, value) in cm {
        fields.insert(key.clone(), core_to_proto_metadata_value(value));
    }
    pb::Metadata { fields }
}

/// Parse a distance metric string into the core enum.
pub fn parse_distance_metric(s: &str) -> Option<DistanceMetricType> {
    match s.to_lowercase().as_str() {
        "cosine" => Some(DistanceMetricType::Cosine),
        "euclidean" => Some(DistanceMetricType::Euclidean),
        "dot_product" | "dotproduct" => Some(DistanceMetricType::DotProduct),
        "manhattan" => Some(DistanceMetricType::Manhattan),
        _ => None,
    }
}

/// Convert a core DistanceMetricType to its string representation.
pub fn distance_metric_to_string(metric: DistanceMetricType) -> String {
    match metric {
        DistanceMetricType::Cosine => "cosine".to_string(),
        DistanceMetricType::Euclidean => "euclidean".to_string(),
        DistanceMetricType::DotProduct => "dot_product".to_string(),
        DistanceMetricType::Manhattan => "manhattan".to_string(),
    }
}
