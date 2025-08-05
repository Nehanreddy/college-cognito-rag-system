import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, FileText, BarChart3, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';

interface Source {
  source: string;
  relevance_score: number;
  semantic_score: number;
  page_number: any;
  chunk_size: number;
  word_count: number;
  document_type: string;
  preview: string;
}

interface QueryStats {
  retrieved_chunks: number;
  unique_sources: number;
  avg_relevance: number;
  coverage_span: string;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  sources?: Source[];
  stats?: QueryStats;
  isLoading?: boolean;
}

const API_BASE_URL = 'http://localhost:8000';

export const ChatBot: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load suggestions on component mount
    fetchSuggestions();
  }, []);

  const fetchSuggestions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/search-suggestions`);
      if (response.ok) {
        const data = await response.json();
        setSuggestions(data.slice(0, 5)); // Show first 5 suggestions
      }
    } catch (error) {
      console.error('Error fetching suggestions:', error);
    }
  };

  const handleSendMessage = async (question: string = inputValue) => {
    if (!question.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: question,
      timestamp: new Date(),
    };

    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'bot',
      content: 'Thinking...',
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          top_k: 10,
          debug: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const botMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'bot',
        content: data.answer,
        timestamp: new Date(),
        sources: data.sources,
        stats: data.query_stats,
      };

      setMessages(prev => prev.slice(0, -1).concat(botMessage));
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your question. Please make sure the backend server is running and try again.',
        timestamp: new Date(),
      };

      setMessages(prev => prev.slice(0, -1).concat(errorMessage));
      
      toast({
        title: "Connection Error",
        description: "Unable to connect to the college information system. Please check if the server is running.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-subtle">
      {/* Header */}
      <div className="bg-gradient-header text-primary-foreground p-4 shadow-medium">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3">
            <Bot className="w-8 h-8" />
            <div>
              <h1 className="text-xl font-bold">College Cognito</h1>
              <p className="text-sm opacity-90">AI-powered college information assistant</p>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <Bot className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h2 className="text-xl font-semibold mb-2">Welcome to College Cognito</h2>
              <p className="text-muted-foreground mb-6">
                Ask me anything about college admissions, fees, facilities, programs, and more!
              </p>
              
              {/* Suggestions */}
              {suggestions.length > 0 && (
                <div className="space-y-3">
                  <p className="text-sm font-medium">Try asking:</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {suggestions.map((suggestion, index) => (
                      <Button
                        key={index}
                        variant="suggestion"
                        size="sm"
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="text-xs"
                      >
                        {suggestion}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t bg-background p-4 shadow-soft">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about the college..."
              disabled={isLoading}
              className="flex-1 bg-chat-input border-chat-input-border"
            />
            <Button 
              onClick={() => handleSendMessage()}
              disabled={isLoading || !inputValue.trim()}
              variant="chat"
              size="icon"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

const MessageBubble: React.FC<{ message: ChatMessage }> = ({ message }) => {
  return (
    <div className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
      {message.type === 'bot' && (
        <div className="w-8 h-8 rounded-full bg-chat-bot-message border border-border flex items-center justify-center mt-1">
          <Bot className="w-4 h-4" />
        </div>
      )}
      
      <div className={`max-w-3xl ${message.type === 'user' ? 'order-first' : ''}`}>
        <div
          className={`p-4 rounded-2xl shadow-soft ${
            message.type === 'user'
              ? 'bg-chat-user-message text-chat-user-message-foreground ml-12'
              : 'bg-chat-bot-message text-chat-bot-message-foreground'
          }`}
        >
          {message.isLoading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Searching through college documents...</span>
            </div>
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}
        </div>

        {/* Sources and Stats */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 space-y-3">
            {/* Stats Summary */}
            {message.stats && (
              <Card className="bg-source-card border-source-card-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" />
                    Query Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <FileText className="w-3 h-3" />
                      {message.stats.retrieved_chunks} chunks from {message.stats.unique_sources} sources
                    </div>
                    <div className="flex items-center gap-1">
                      <BarChart3 className="w-3 h-3" />
                      {(message.stats.avg_relevance * 100).toFixed(1)}% avg relevance
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Coverage: {message.stats.coverage_span}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Sources */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-muted-foreground">Sources:</h4>
              {message.sources.slice(0, 3).map((source, index) => (
                <SourceCard key={index} source={source} />
              ))}
              {message.sources.length > 3 && (
                <p className="text-xs text-muted-foreground">
                  and {message.sources.length - 3} more sources...
                </p>
              )}
            </div>
          </div>
        )}

        <div className="text-xs text-muted-foreground mt-2">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>

      {message.type === 'user' && (
        <div className="w-8 h-8 rounded-full bg-chat-user-message text-chat-user-message-foreground flex items-center justify-center mt-1">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  );
};

const SourceCard: React.FC<{ source: Source }> = ({ source }) => {
  return (
    <Card className="bg-source-card border-source-card-border">
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <FileText className="w-3 h-3 text-muted-foreground" />
              <span className="text-sm font-medium truncate">
                {source.source.split('/').pop() || source.source}
              </span>
              {source.page_number !== 'N/A' && (
                <Badge variant="outline" className="text-xs">
                  Page {source.page_number}
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground line-clamp-2">
              {source.preview}
            </p>
          </div>
          <div className="text-right">
            <div className="text-xs font-medium">
              {(source.relevance_score * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">relevance</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};