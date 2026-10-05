from rest_framework import serializers

class AskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=500)

class MatchSerializer(serializers.Serializer):
    source = serializers.CharField(required=False)
    chunk_id = serializers.IntegerField(required=False)
    snippet = serializers.CharField(required=False)
    confidence = serializers.FloatField(required=False)

class AskResponseSerializer(serializers.Serializer):
    question = serializers.CharField()
    answer = serializers.CharField()
    contexts = serializers.ListField(child=serializers.CharField(), required=False)
    matches = MatchSerializer(many=True, required=False)
    latency_ms = serializers.IntegerField(required=False)
    metadata = serializers.DictField(required=False)  # Include metadata with avg_similarity, model, etc.
